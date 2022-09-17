import os
from glob import glob

from typing import Tuple, Dict, List, Optional
import argparse
import json
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, resnet50, vgg16_bn

from tqdm import trange

from al import ActivePool
from al.methods import NAME_TO_CLS
from utils import set_seed, Tracker, write_json
from arguments import *


def create_and_parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser("CIFAR10-ENSEMBLE")

    parser.add_argument('-f', '--file', type=str, required=False)

    parser.add_argument('--run_name',     type=str, default='cifar10-ensemble')
    parser.add_argument('--project',      type=str, default='al_swa')
    parser.add_argument('--save_path',    type=str, default='saved/')
    parser.add_argument('--dataset_path', type=str, default='datasets/cifar10')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arch', type=str, default="resnet18", choices=["resnet18", "resnet50", "vgg16"])
    
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--resume_from', type=str, required=None, help='Resume AL from the saved path.')

    parser = add_training_args(parser)
    parser = add_query_args(parser)

    args = parser.parse_args()

    return args


def init_model_and_optimizer(config, num_classes:int=10) -> Tuple[nn.Module, optim.Optimizer]:

    if config.arch == "resnet18":
        model = resnet18(pretrained=False, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
    elif config.arch == "resnet50":
        model = resnet50(pretrained=False, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, 1, 1)
        model.maxpool = nn.Identity()
    elif config.arch == "vgg16":
        model = vgg16_bn(pretrained=False)
        model.avgpool = nn.Identity()
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    else:
        raise ValueError
        
    if config.optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)
    elif config.optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise ValueError
    
    return model, optimizer


def create_scheduler(config, optimizer: optim.Optimizer, steps_per_epoch: int) -> LambdaLR:
    if config.lr_scheduler_type == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            config.learning_rate*config.lr_scheduler_param,
            epochs=config.num_epochs if not config.start_swa_at_end else config.swa_start,
            steps_per_epoch=steps_per_epoch,
        )
    elif config.lr_scheduler_type == "none":
        scheduler = LambdaLR(optimizer, lambda epoch: 1.0)
    else:
        raise ValueError

    return scheduler


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, lr_scheduler: Optional[LambdaLR]=None, device: Optional[torch.device]=None) -> float:

    train_loss = Tracker("train_loss")
    loss_fn = nn.CrossEntropyLoss()

    if lr_scheduler is not None:
        if isinstance(lr_scheduler, OneCycleLR):
            lr_update_strategy = "step"
        else:
            lr_update_strategy = "epoch"
    else:
        lr_update_strategy = "none"

    model.train()
    for imgs, lbls in dataloader:

        if device is not None:
            imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, lbls)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss.update(loss.item(), imgs.size(0))

        if lr_update_strategy == "step":
            lr_scheduler.step()

    if lr_update_strategy == "epoch":
        lr_scheduler.step()

    return train_loss.get()


@torch.no_grad()
def eval(model: nn.Module, dataloader: DataLoader, device: Optional[torch.device]=None) -> Tuple[float, torch.Tensor]:
    
    results = predict(model, dataloader, device)
    acc = accuracy_score(results['targets'].numpy(), results['preds'].numpy())

    return acc, results['logits']


@torch.no_grad()
def predict(model: nn.Module, dataloader: DataLoader, device: Optional[torch.device]=None) -> Dict[str, torch.TensorType]:
    all_preds   = []
    all_targets = []
    all_logits  = []
    
    model.eval()
    for imgs, lbls in dataloader:

        if device is not None:
            imgs, lbls = imgs.to(device), lbls.to(device)
        
        logits = model(imgs)
        preds = torch.argmax(logits, dim=-1)

        all_logits.append(logits.cpu())
        all_targets.append(lbls.cpu())
        all_preds.append(preds.cpu())

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_logits  = torch.cat(all_logits,  dim=0)

    return {
        "preds": all_preds,
        "targets": all_targets,
        "logits": all_logits
    }


def test_ensemble(checkpoints: List[str], targets: List[int], model: nn.Module, dataloader: DataLoader, device: Optional[torch.device]=None):

    all_logits = []
    accs = []

    for idx, ckpt in enumerate(checkpoints):
        state_dict = torch.load(ckpt, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict)
        model.to(device)
        acc, logits = eval(model, dataloader, device)
        accs.append(acc)
        print(f"model_{idx} test accuracy: {acc*100:.2f}%", end=" ")
        all_logits.append(logits.detach().cpu().unsqueeze(1))
    print()

    all_logits = torch.cat(all_logits, dim=1)
    ens_logits = torch.mean(all_logits, dim=1)
    ens_preds  = torch.argmax(ens_logits, dim=-1).numpy()

    ens_acc    = accuracy_score(ens_preds, np.asarray(targets))
    mean_acc   = np.mean(accs)

    return ens_acc, mean_acc


def main(config):

    if config.seed is not None:
        set_seed(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2470, 0.2435, 0.2616]

    train_transform = T.Compose([
        T.RandomHorizontalFlip(), 
        T.ToTensor(), 
        T.Normalize(mean, std)
    ])
    test_transform  = T.Compose([
        T.ToTensor(), 
        T.Normalize(mean, std)
    ])

    train_set = CIFAR10(root=config.dataset_path, train=True,  download=True, transform=train_transform)
    query_set = CIFAR10(root=config.dataset_path, train=True,  download=True, transform=test_transform)
    test_set  = CIFAR10(root=config.dataset_path, train=False, download=True, transform=test_transform)

    pool = ActivePool(train_set=train_set, query_set=query_set, test_set=test_set, batch_size=config.batch_size)

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
            if ens == 0:
                model, optimizer = init_model_and_optimizer(config)
                init_ckpt_file = os.path.join(config.save_path, f"episode{episode}_init.ckpt")
                torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, init_ckpt_file)
                print(f"Initial checkpoint saved to {init_ckpt_file}")
            else:
                ckpt = torch.load(init_ckpt_file)
                msg = model.load_state_dict(ckpt['state_dict'], strict=True)
                optimizer.load_state_dict(ckpt['optimizer'])
                print(f"Checkpoint loaded from {init_ckpt_file} -- {msg}")
            
            scheduler = create_scheduler(config, optimizer, len(pool.get_labeled_dataloader(drop_last=True)))

            sampler.update_model(model) # this updates the reference to the model.
            model.to(device)
            
            tbar = trange(1, config.num_epochs+1, disable=config.disable_tqdm)
            max_acc, eval_acc = 0.0, 0.0

            for epoch in tbar:
                model.train()
                train_loss = train_epoch(model, pool.get_labeled_dataloader(), optimizer, scheduler, device)
                
                if epoch % config.eval_every == 0:
                    model.eval()
                    eval_acc = eval(model, pool.get_eval_dataloader(), device)

                tbar.set_description(f"train loss {train_loss():.3f}, eval acc {eval_acc*100:.2f}")

            ckpt_file = os.path.join(episode_save_path, f"member_{ens}.ckpt")
            torch.save({"state_dict": model.state_dict()}, ckpt_file)
            checkpoints.append(ckpt_file)
        
        ens_acc, mean_acc = test_ensemble(checkpoints, test_set.targets, model, pool.get_test_dataloader(), device)
        print(f"Episode {episode} num_models: {len(checkpoints)} -- max eval acc: {max_acc*100:.2f}, test ens_acc: {ens_acc*100:.2f}, mean_acc: {mean_acc*100:.2f}")

        query_result = sampler(checkpoints=checkpoints)
        queried_ids = pool.convert_to_original_ids(query_result.indices)
        write_json(queried_ids, os.path.join(episode_save_path, f"queried_ids.json"))

        metrics = {
            "episode": episode,
            "num_ensembles": len(checkpoints),
            "eval/acc": eval_acc,
            "eval/max_acc": max_acc,
            "test/ens_acc": ens_acc,
            "test/mean_acc": mean_acc,
            "episode/indicies": queried_ids,
            "episode/scores": query_result.scores,
            "episode/num_labeled": len(pool.get_labeled_ids()),
        }
        pool.update(query_result)

        write_json(metrics, os.path.join(episode_save_path, f"result.json"))
        episode_results.append(metrics)

        write_json(episode_results, os.path.join(config.save_path, "results.json"))


if __name__ == '__main__':

    args = create_and_parse_args()

    if args.file is not None:
        with open(args.file, "r") as f:
            args_dict = json.load(args.file)
        args_dict.drop('file')
        args.__dict__.update(args_dict)

    print(vars(args))

    args.run_name = f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.save_path = os.path.join(args.save_path, args.run_name)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    write_json(vars(args), os.path.join(args.save_path, "config.json"))

    main(args)