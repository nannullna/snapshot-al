from typing import List, Optional, Tuple, Dict
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
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR, CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from tqdm import trange

from al import ActivePool
from al.methods import NAME_TO_CLS, LabeledRandomSampling
from utils import set_seed, write_json

from arguments import *
from commons import (
    create_active_pool, init_model_and_optimizer, init_model, init_optimizer, \
    train_epoch, train_epoch_regul, eval, predict, test_ensemble
)


def create_and_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("CIFAR-Snapshot with FT")

    parser.add_argument('-f', '--file', type=str, required=False)

    parser.add_argument('--run_name',     type=str, default='cifar-snapshot')
    parser.add_argument('--save_path',    type=str, default='saved/')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--dataset_path', type=str, default='datasets')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arch', type=str, default='resnet18', choices=["resnet18", "resnet50", "vgg16"])

    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--resume_from', type=str, required=None, help='Resume AL from the saved path.')

    parser.add_argument('--regul', action='store_false')
    parser.add_argument('--lamb', type=float, required=None, default=1e-2)

    parser = add_training_args(parser)
    parser = add_snapshot_args(parser)
    parser = add_query_args(parser)

    args = parser.parse_args()

    return args


def create_scheduler(config, optimizer: optim.Optimizer, steps_per_epoch: int) -> LambdaLR:
    if config.lr_scheduler_type == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            config.learning_rate*config.lr_scheduler_param,
            epochs=config.num_epochs if not config.start_snapshot_at_end else config.snapshot_start,
            steps_per_epoch=steps_per_epoch,
        )
    elif config.lr_scheduler_type in ["none", "constant"]:
        scheduler = LambdaLR(optimizer, lambda epoch: 1.0)
    else:
        raise ValueError

    return scheduler


def create_swa_model_and_scheduler(config, model: nn.Module, optimizer: optim.Optimizer, save_interval: int) -> Tuple[AveragedModel, LambdaLR]:
    swa_model = AveragedModel(model)

    if config.swa_scheduler_type == "constant":
        swa_scheduler = SWALR(
            optimizer, swa_lr=config.learning_rate*config.snapshot_lr_multiplier, 
            anneal_epochs=config.snapshot_anneal_epochs, anneal_strategy="cos"
        )
    elif config.swa_scheduler_type == "cosine":
        swa_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=save_interval, T_mult=1, eta_min=1e-5)
        swa_scheduler.base_lrs = [config.snapshot_lr_multiplier*config.learning_rate \
            for base_lr in swa_scheduler.base_lrs]
    elif config.swa_scheduler_tyhpe == "none":
        swa_scheduler = LambdaLR(optimizer, lambda epoch: 1)
    else:
        raise ValueError

    return swa_model, swa_scheduler
    

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

    num_epochs_, snapshot_start_ = config.num_epochs, config.snapshot_start
    model = init_model(config)
    for episode in range(last_episode+1, config.num_episodes+1):

        checkpoints: List[str] = []

        episode_save_path = os.path.join(config.save_path, f"episode_{episode}")
        os.makedirs(episode_save_path)

        if episode == config.num_episodes:
            print("============ Start from scratch ============")
            model, optimizer = init_model_and_optimizer(config)

            config.num_epochs = num_epochs_
            config.snapshot_start = snapshot_start_

            dataloader = pool.get_labeled_dataloader(drop_last=True)

        else:
            print("============ Fine tuning ============")
            optimizer = init_optimizer(config, model)

            config.num_epochs = config.ft_epochs
            config.snapshot_start = config.ft_snapshot_start

            dataloader = pool.get_active_dataloader(drop_last=True)

        save_interval = (config.num_epochs - config.snapshot_start) // config.num_ensembles
        save_at = [config.num_epochs - i*save_interval for i in range(config.num_ensembles)][::-1]
        print(f"Total of {len(save_at)} models expected at {save_at}.")

        scheduler = create_scheduler(config, optimizer, len(pool.get_labeled_dataloader(drop_last=False)))
        swa_model, swa_scheduler = create_swa_model_and_scheduler(config, model, optimizer, save_interval)

        sampler.update_model(model) # this updates the reference to the model.
        model.to(device)

        tbar = trange(1, config.num_epochs+1, disable=config.disable_tqdm)
        max_acc = 0.0
        regul_mode = False

        print(len(dataloader.dataset))
        for epoch in tbar:

            model.train()
            train_loss = train_epoch_regul(model, swa_model, dataloader, optimizer, scheduler if epoch <= config.swa_start else None, device, regul_mode, config.lamb)

            if epoch > config.snapshot_start:
                regul_mode = config.regul
                swa_model.update_parameters(model)
                swa_scheduler.step()
                
                if epoch in save_at:
                    ckpt_file_name = os.path.join(episode_save_path, f"epoch_{epoch}.ckpt")
                    torch.save({"state_dict": model.state_dict()}, ckpt_file_name)
                    checkpoints.append(ckpt_file_name)

            if epoch % config.eval_every == 0:
                model.eval()
                eval_results = eval(model, pool.get_eval_dataloader(num_workers=config.num_workers, pin_memory=True), device)
                eval_acc = eval_results['acc']
                if eval_acc > max_acc:
                    max_acc = eval_acc
                tbar.set_description(f"train loss {train_loss:.3f}, eval acc {eval_acc*100:.2f}")

        swa_model.to(device)
        update_bn(pool.get_labeled_dataloader(num_workers=config.num_workers, pin_memory=True), swa_model, device)
        swa_ckpt_name = os.path.join(episode_save_path, f"swa_model.ckpt")
        torch.save({"state_dict": swa_model.state_dict()}, swa_ckpt_name)

        swa_model.eval()
        swa_results = eval(swa_model, pool.get_test_dataloader(num_workers=config.num_workers, pin_memory=True), device)
        swa_acc = swa_results['acc']
        print(f"Episode {episode} num_models: {len(checkpoints)} -- max eval acc: {max_acc*100:.2f}, test acc: {swa_acc*100:.2f}")

        ens_metrics = test_ensemble(checkpoints, model, pool.get_test_dataloader(num_workers=config.num_workers, pin_memory=True), device)
        print(f"ens acc: {ens_metrics['ens/acc']*100:.2f}, mean acc: {ens_metrics['ens/mean_acc']*100:.2f}")

        if episode >= 4:
            random_size = 2000
        else:
            random_size = config.query_size * episode
        random_sampler = LabeledRandomSampling(model=None, pool=pool, size=random_size, device=device)
        random_sampler.update_model(model)

        random_query_result = random_sampler(checkpoints=checkpoints, swa_checkpoint=swa_ckpt_name)
        random_queried_ids  = pool.convert_to_original_ids_labeled(random_query_result.indices)

        query_result = sampler(checkpoints=checkpoints, swa_checkpoint=swa_ckpt_name)
        queried_ids  = pool.convert_to_original_ids(query_result.indices)
        merge_queried_ids = list(set(queried_ids + random_queried_ids))
        write_json(merge_queried_ids, os.path.join(episode_save_path, f"queried_ids.json"))
        pool.update(query_result)

        pool.active_data.indices = list(set(queried_ids + random_queried_ids))

        metrics = {
            "episode": episode,
            "num_ensembles": len(checkpoints),
            "eval/acc": eval_acc,
            "eval/max_acc": max_acc,
            "test/swa_acc": swa_results['acc'],
            "test/swa_nll": swa_results['nll'],
            "test/swa_ece": swa_results['ece'],
            "test/swa_top5": swa_results['top5'],
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