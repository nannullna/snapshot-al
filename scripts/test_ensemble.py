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
    create_active_pool, init_model, test_ensemble
)


def create_and_parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser("Retrain")

    parser.add_argument('-f', '--file', type=str, required=False)

    parser.add_argument('--run_name',     type=str, default='test')
    parser.add_argument('--save_path',    type=str, required=False)
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tiny'])
    parser.add_argument('--dataset_path', type=str, default='datasets')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arch', type=str, default="resnet18", choices=["resnet18", "resnet50", "vgg16", "densenet121"])
    
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--resume_from', type=str, required=None, help='Resume AL from the saved path.')

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

    checkpoints: List[str] = sorted(glob(os.path.join(config.resume_from, '*.ckpt')))

    model = init_model(config)
    
    print()
    ens_metrics = test_ensemble(checkpoints, model, pool.get_test_dataloader(num_workers=config.num_workers, pin_memory=True), device)
    print(f"num_models: {len(checkpoints)} -- test ens_acc: {ens_metrics['ens/acc']*100:.2f}, mean_acc: {ens_metrics['ens/mean_acc']*100:.2f}")

    metrics = {
        "num_ensembles": len(checkpoints),
    }
    metrics.update(ens_metrics)

    write_json(metrics, os.path.join(config.save_path, f"test_result.json"))


if __name__ == '__main__':

    args = create_and_parse_args()

    if args.file is not None:
        with open(args.file, "r") as f:
            args_dict = json.load(f)
        args.__dict__.update(args_dict)

    print(vars(args))

    args.save_path = args.save_path or args.resume_from
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    print(f"Experiment results will be saved to {args.save_path}")

    write_json(vars(args), os.path.join(args.save_path, "config.json"))

    main(args)