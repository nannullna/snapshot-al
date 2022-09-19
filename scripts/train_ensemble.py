from typing import Any, Tuple, Dict, List, Optional
import os
import sys
sys.path.append('.')
sys.path.append('..')
from glob import glob

import argparse
import json
from datetime import datetime

import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

from tqdm import trange

from al.methods import NAME_TO_CLS
from utils import set_seed, write_json

from arguments import *
from commons import (
    create_active_pool, init_model_and_optimizer, \
    train_epoch, eval, predict, test_ensemble
)


def create_and_parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser("CIFAR-DE")

    parser.add_argument('-f', '--file', type=str, required=False)

    parser.add_argument('--run_name',     type=str, default='cifar-de')
    parser.add_argument('--save_path',    type=str, default='saved/')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--dataset_path', type=str, default='datasets')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arch', type=str, default="resnet18", choices=["resnet18", "resnet50", "vgg16"])
    
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--resume_from', type=str, required=None, help='Resume AL from the saved path.')

    parser = add_training_args(parser)
    parser = add_query_args(parser)

    args = parser.parse_args()

    return args


def create_scheduler(config, optimizer: optim.Optimizer, steps_per_epoch: int) -> LambdaLR:
    if config.lr_scheduler_type == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            config.learning_rate*config.lr_scheduler_param,
            epochs=config.num_epochs,
            steps_per_epoch=steps_per_epoch,
        )
    elif config.lr_scheduler_type in ["none", "constant"]:
        scheduler = LambdaLR(optimizer, lambda epoch: 1.0)
    else:
        raise ValueError

    return scheduler


def main(config):

    if config.seed is not None:
        set_seed(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pool = create_active_pool(config)

    episode_results = []

    if config.resume_from is None:

        eval_sampler = NAME_TO_CLS[config.eval_query_type](model=None, pool=pool, size=config.eval_query_size, device=device)
        init_sampler = NAME_TO_CLS[config.init_query_type](model=None, pool=pool, size=config.init_query_size, device=device)

        episode_save_path = os.path.join(config.save_path, f"episode_{0}")
        os.makedirs(episode_save_path)

        eval_query_result = eval_sampler()
        write_json(pool.convert_to_original_ids(eval_query_result.indices), os.path.join(episode_save_path, f"eval_queried_ids.json"))
        pool.update_eval(eval_query_result)

        query_result = init_sampler()
        write_json(pool.convert_to_original_ids(query_result.indices), os.path.join(episode_save_path, f"queried_ids.json"))
        pool.update(query_result)
        print(pool)

        last_episode = 0
        metrics = {
            "episode": 0,
            "episode/indicies": pool.get_labeled_ids(),
        }
        write_json(metrics, os.path.join(episode_save_path, f"result.json"))

        episode_results.append(metrics)


    else:

        eval_queried_files = glob(os.path.join(config.resume_from, "episode_*", "eval_queried_ids.json"))
        eval_queried_ids = []
        for queried_file in eval_queried_files:
            with open(queried_file, "r") as f:
                eval_queried_ids.extend(json.load(f))
        pool.update_eval(eval_queried_ids, original_ids=True)

        queried_files = glob(os.path.join(config.resume_from, "episode_*", "queried_ids.json"))
        queried_ids = []
        for queried_file in queried_files:
            with open(queried_file, "r") as f:
                queried_ids.extend(json.load(f))
        pool.update(queried_ids, original_ids=True)

        episodes = [os.path.split(f)[0] for f in queried_files]
        episodes = [int(ep[ep.find("episode_")+8:]) for ep in episodes]
        last_episode = max(episodes)

        print(pool)
    
    sampler = NAME_TO_CLS[config.query_type](model=None, pool=pool, size=config.query_size, device=device)

    for episode in range(last_episode+1, config.num_episodes+1):

        checkpoints: List[str] = []

        episode_save_path = os.path.join(config.save_path, f"episode_{episode}")
        os.makedirs(episode_save_path)

        for ens in range(config.num_ensembles):
            model, optimizer = init_model_and_optimizer(config)
            scheduler = create_scheduler(config, optimizer, len(pool.get_labeled_dataloader(drop_last=False)))

            sampler.update_model(model) # this updates the reference to the model.
            model.to(device)
            
            tbar = trange(1, config.num_epochs+1, disable=config.disable_tqdm)
            max_acc, eval_acc = 0.0, 0.0

            for epoch in tbar:
                model.train()
                train_loss = train_epoch(model, pool.get_labeled_dataloader(num_workers=config.num_workers, pin_memory=True), optimizer, scheduler, device)
                
                if epoch % config.eval_every == 0:
                    model.eval()
                    eval_results = eval(model, pool.get_eval_dataloader(num_workers=config.num_workers, pin_memory=True), device)
                    eval_acc = eval_results['acc']

                tbar.set_description(f"train loss {train_loss:.3f}, eval acc {eval_acc*100:.2f}")

            ckpt_file = os.path.join(episode_save_path, f"member_{ens}.ckpt")
            torch.save({"state_dict": model.state_dict()}, ckpt_file)
            checkpoints.append(ckpt_file)
        
        ens_metrics = test_ensemble(checkpoints, model, pool.get_test_dataloader(num_workers=config.num_workers, pin_memory=True), device)
        print(f"Episode {episode} num_models: {len(checkpoints)} -- max eval acc: {max_acc*100:.2f}, test ens_acc: {ens_metrics['ens/acc']*100:.2f}, mean_acc: {ens_metrics['ens/mean_acc']*100:.2f}")

        query_result = sampler(checkpoints=checkpoints)
        queried_ids = pool.convert_to_original_ids(query_result.indices)
        write_json(queried_ids, os.path.join(episode_save_path, f"queried_ids.json"))
        pool.update(query_result)

        metrics = {
            "episode": episode,
            "num_ensembles": len(checkpoints),
            "eval/acc": eval_acc,
            "eval/max_acc": max_acc,
            "episode/indices": queried_ids,
            "episode/scores": query_result.scores,
            "episode/num_labeled": len(pool.get_labeled_ids()),
        }
        metrics.update(ens_metrics)

        write_json(metrics, os.path.join(episode_save_path, f"result.json"))
        episode_results.append(metrics)

        write_json(episode_results, os.path.join(config.save_path, "results.json"))


if __name__ == '__main__':

    args = create_and_parse_args()

    if args.file is not None:
        with open(args.file, "r") as f:
            args_dict = json.load(f)
        args.__dict__.update(args_dict)

    print(vars(args))

    args.run_name = f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.save_path = os.path.join(args.save_path, args.run_name)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    print(f"Experiment results will be saved to {args.save_path}")

    write_json(vars(args), os.path.join(args.save_path, "config.json"))

    main(args)