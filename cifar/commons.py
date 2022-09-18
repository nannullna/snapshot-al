from typing import Tuple, List, Dict, Any, Optional
import os

import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import resnet18, resnet50, vgg16_bn

from al import ActivePool
from utils import Tracker
from evaluate import ece_loss, nll


def create_active_pool(config) -> ActivePool:
    
    if config.dataset_name == 'cifar10':

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

        train_set = CIFAR10(root=os.path.join(config.dataset_path, config.dataset_name), train=True,  download=True, transform=train_transform)
        query_set = CIFAR10(root=os.path.join(config.dataset_path, config.dataset_name), train=True,  download=True, transform=test_transform)
        test_set  = CIFAR10(root=os.path.join(config.dataset_path, config.dataset_name), train=False, download=True, transform=test_transform)

    elif config.dataset_name == 'cifar100':

        mean = [0.5071, 0.4865, 0.4409]
        std  = [0.2673, 0.2564, 0.2762]

        train_transform = T.Compose([
            T.RandomHorizontalFlip(), 
            T.ToTensor(), 
            T.Normalize(mean, std)
        ])
        test_transform  = T.Compose([
            T.ToTensor(), 
            T.Normalize(mean, std)
        ])

        train_set = CIFAR100(root=os.path.join(config.dataset_path, config.dataset_name), train=True,  download=True, transform=train_transform)
        query_set = CIFAR100(root=os.path.join(config.dataset_path, config.dataset_name), train=True,  download=True, transform=test_transform)
        test_set  = CIFAR100(root=os.path.join(config.dataset_path, config.dataset_name), train=False, download=True, transform=test_transform)

    pool = ActivePool(train_set=train_set, query_set=query_set, test_set=test_set, batch_size=config.batch_size)
    
    return pool


def init_model_and_optimizer(config) -> Tuple[nn.Module, optim.Optimizer]:

    num_classes = 10 if config.dataset_name == 'cifar10' else 100

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


def calc_metrics(eval_results: Dict[str, torch.Tensor]) -> Dict[str, float]:
    probs_np = F.softmax(eval_results['logits'], dim=-1).numpy()
    targets_np = eval_results['targets'].numpy()
    preds_np = eval_results['preds'].numpy()

    acc_ = accuracy_score(targets_np, preds_np)
    nll_ = nll(probs_np, targets_np)
    ece_ = ece_loss(probs_np, targets_np, n_bins=10)

    return {"acc": float(acc_), "nll": float(nll_), "ece": float(ece_)}


@torch.no_grad()
def eval(model: nn.Module, dataloader: DataLoader, device: Optional[torch.device]=None) -> Dict[str, Any]:
    
    results = predict(model, dataloader, device)

    metrics = calc_metrics(results)
    results.update(metrics)

    return results


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


def test_ensemble(checkpoints: List[str], model: nn.Module, dataloader: DataLoader, device: Optional[torch.device]=None) -> Dict[str, float]:

    all_logits = []
    accs = []

    for idx, ckpt in enumerate(checkpoints):
        state_dict = torch.load(ckpt, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict)
        model.to(device)
        eval_results = eval(model, dataloader, device)
        accs.append(eval_results['acc'])
        print(f"model_{idx} test accuracy: {eval_results['acc']*100:.2f}%", end=" ")
        all_logits.append(eval_results['logits'].detach().cpu().unsqueeze(1))
    print()

    all_logits = torch.cat(all_logits, dim=1)
    ens_logits = torch.mean(all_logits, dim=1)
    ens_preds  = torch.argmax(ens_logits, dim=-1)

    ens_results = {'logits': ens_logits, 'targets': eval_results['targets'], 'preds': ens_preds}
    ens_metrics = calc_metrics(ens_results)
    ens_metrics = {f"ens_{k}": v for k, v in ens_metrics.items()}
    ens_metrics['mean_acc'] = float(np.mean(accs))
    ens_metrics['std_acc']  = float(np.std(accs))

    return ens_metrics