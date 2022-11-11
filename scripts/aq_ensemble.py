from typing import List, Optional, Tuple, Dict
import os
import sys
sys.path.append(".")

import argparse
import json
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR, MultiStepLR, CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder
from torchvision.models import resnet18, resnet50, vgg16_bn

from tqdm import tqdm, trange

from utils import Tracker, set_seed

from arguments import *
from commons import (
    create_active_pool, init_model_and_optimizer, \
    train_epoch, eval, predict, test_ensemble
)


def create_scheduler(config, optimizer: optim.Optimizer, steps_per_epoch: int) -> LambdaLR:
    if config.lr_scheduler_type == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            config.learning_rate*config.lr_scheduler_param,
            epochs=config.num_epochs,
            steps_per_epoch=steps_per_epoch,
        )
    elif config.lr_scheduler_type == "none":
        scheduler = LambdaLR(optimizer, lambda epoch: 1.0)
    elif config.lr_scheduler_type in ["none", "constant"]:
        scheduler = LambdaLR(optimizer, lambda epoch: 1.0)
    elif config.lr_scheduler_type == "step":
        first_milestone  = int(config.num_epochs * 0.5)
        second_milestone = int(config.num_epochs * 0.75)
        scheduler = MultiStepLR(optimizer, milestones=[first_milestone, second_milestone], gamma=config.lr_scheduler_param)
    else:
        raise ValueError

    return scheduler


def main(config):

    print(config.file)
    print(config.result)

    if config.seed is not None:
        set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    if config.dataset_name == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std  = [0.2470, 0.2435, 0.2616]
        input_size = 32

        train_transform = T.Compose([
            T.RandomHorizontalFlip(), 
            T.ToTensor(), 
            T.Normalize(mean, std)
        ])
        test_transform  = T.Compose([
            T.ToTensor(), 
            T.Normalize(mean, std)
        ])

    elif config.dataset_name == "cifar100":

        mean = [0.4914, 0.4822, 0.4465]
        std = [0.247 , 0.2435, 0.2616]
        # mean = [0.5071, 0.4865, 0.4409]
        # std  = [0.2673, 0.2564, 0.2762]
        root = os.path.join(config.dataset_path, config.dataset_name)

        train_transform = T.Compose([
            T.RandomCrop(32, 4),
            T.RandomHorizontalFlip(), 
            T.ToTensor(), 
            T.Normalize(mean, std)
        ])
        test_transform  = T.Compose([
            T.ToTensor(), 
            T.Normalize(mean, std)
        ])
        print('load cifar100 complete')
        
    elif config.dataset_name == 'tiny':

        def create_eval_img_folder(dataset_path: str):

            val_dir = os.path.join(dataset_path, 'val')
            img_dir = os.path.join(val_dir, 'images')

            fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
            data = fp.readlines()
            eval_img_dict = OrderedDict()
            for line in tqdm(data):
                words = line.split('\t')
                eval_img_dict[words[0]] = words[1]
            fp.close()

            # Create folder if not present and move images into proper folders
            for img, folder in tqdm(eval_img_dict.items()):
                newpath = os.path.join(img_dir, folder)
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                if os.path.exists(os.path.join(img_dir, img)):
                    os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        size = 64
        
        train_dir = os.path.join(config.dataset_path, 'train')
        val_dir   = os.path.join(config.dataset_path, 'val', 'images')

        # de-facto standard augmentations for tiny-imagenet in literature
        train_transform = T.Compose([
            T.RandomCrop(size, 4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

        create_eval_img_folder(config.dataset_path)

        train_set = ImageFolder(train_dir, transform=train_transform)
        test_set  = ImageFolder(val_dir,   transform=test_transform)

        
    num_episode = args.episode
    
    # Gathering queried examples from exp_path's "episode*_queried_ids.json"
    queried_ids, total_queried_ids = [], []
    queried_file = config.result
    with open(queried_file, mode='r', encoding='utf-8') as f:
        queried_ids.extend(json.load(f))

    for i in range(num_episode):
        # print(queried_ids[i].keys())
        total_queried_ids += queried_ids[i]['episode/indices']

    print(f"Length of labeled set {len(total_queried_ids)}, eval set ")

    # first_ids, second_ids = [], []

    # with open('query/cifar100/de/maxentropy/seed42/1st_results.json', mode='r', encoding='utf-8') as f:
    #     first_ids.extend(json.load(f))

    # for i in range(7):
    #     # print(queried_ids[i].keys())
    #     total_queried_ids += first_ids[i]['episode/indices']

    # with open('query/cifar100/de/maxentropy/seed42/results.json', mode='r', encoding='utf-8') as f:
    #     second_ids.extend(json.load(f))

    # for i in range(0, num_episode-7):
    #     # print(queried_ids[i].keys())
    #     total_queried_ids += second_ids[i]['episode/indices']

    # print(f"Length of labeled set {len(total_queried_ids)}, eval set ")


    if config.dataset_name == "cifar10":
        raw_datasets = CIFAR10(root=config.dataset_path, train=True,  download=True, transform=train_transform)
        test_set  = CIFAR10(root=config.dataset_path, train=False, download=True, transform=test_transform)

    elif config.dataset_name == "cifar100":
        raw_datasets = CIFAR100(root=config.dataset_path, train=True,  download=True, transform=train_transform)
        test_set  = CIFAR100(root=config.dataset_path, train=False, download=True, transform=test_transform)

    elif config.dataset_name == "tiny":
        raw_datasets = ImageFolder(train_dir, transform=train_transform)
        test_set  = ImageFolder(val_dir,   transform=test_transform)

    train_set = Subset(raw_datasets, total_queried_ids)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=config.batch_size, shuffle=False)

    checkpoints: List[str] = []

    for ens in range(config.num_ensembles):
    
        model, optimizer = init_model_and_optimizer(config)
        scheduler = create_scheduler(config, optimizer, len(train_loader))
        
        model.to(device)

        tbar = trange(1, config.num_epochs+1, disable=config.disable_tqdm)
        max_acc = 0.0

        for epoch in tbar:

            model.train()
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)

            # if epoch % config.eval_every == 0:
            #     model.eval()
            #     eval_acc, _ = eval(model, eval_loader, device)
            #     if eval_acc > max_acc:
            #         max_acc = eval_acc
            if epoch % 50 == 0 :
                print(f'cur ens / cur epoch : {ens}/{epoch} ')
                
            tbar.set_description(f"train loss {train_loss:.3f}, eval acc {0*100:.2f}")

        ckpt_file = os.path.join(config.save_path, f"model{ens}.ckpt")
        torch.save({"state_dict": model.state_dict()}, ckpt_file)
        checkpoints.append(ckpt_file)
    
    ens_metrics = test_ensemble(checkpoints, model, test_loader, device)
    print(ens_metrics)
    print(f"num_models: {len(checkpoints)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DE-aq") 

    # parser.add_argument('-f', '--file', type=str, required=False, default='query/cifar10/r18/single/entropy/config.json')
    # parser.add_argument('--result', type=str, required=False, default='query/cifar10/r18/single/entropy/results.json')
    # parser.add_argument('-f', '--file', type=str, required=False, default='query/tiny/se/vr/config.json')
    # parser.add_argument('--result', type=str, required=False, default='query/tiny/se/vr/results.json')
    # parser.add_argument('-f', '--file', type=str, required=False, default='query/cifar100/single/config.json')
    # parser.add_argument('--result', type=str, required=False, default='query/cifar100/single/results.json')
    # parser.add_argument('-f', '--file', type=str, required=False, default='query/cifar100/mc/vr/config.json')
    # parser.add_argument('--result', type=str, required=False, default='query/cifar100/mc/vr/results.json')
    # parser.add_argument('-f', '--file', type=str, required=False, default='query/tiny/mcdropout/vr/config.json')
    # parser.add_argument('--result', type=str, required=False, default='query/tiny/mcdropout/vr/results.json')
    # parser.add_argument('-f', '--file', type=str, required=False, default='query/cifar10/r18/mcdropout/bald/config.json')
    # parser.add_argument('--result', type=str, required=False, default='query/cifar10/r18/mcdropout/bald/results.json')


    parser.add_argument('-f', '--file', type=str, required=False, default='query/cifar100/se/bald/seed42/config.json')
    parser.add_argument('--result', type=str, required=False, default='query/cifar100/se/bald/seed42/results.json')


    parser.add_argument('--episode', type=int, required=True)

    parser.add_argument('--run_name',     type=str, default='DE-aq')
    parser.add_argument('--project',      type=str, default='al_train')
    parser.add_argument('--save_path',    type=str, default='saved/aq/')
    parser.add_argument('--dataset_path', type=str, default='datasets/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--arch', type=str, default='resnet18')

    # parser.add_argument('--exp_path',     type=str, required=True)
    parser.add_argument('--until', type=int, required=False)


    parser = add_training_args(parser)
    parser.add_argument('--num_ensembles', type=int, default=5)

    args = parser.parse_args()
    # print(vars(args))

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

    wandb.init(name=args.run_name, config=args, project=args.project, mode="disabled")
    config = wandb.config

    main(config)