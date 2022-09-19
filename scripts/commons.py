from typing import Tuple, List, Dict, Any, Optional
from collections import OrderedDict
import os

import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.models import resnet18, resnet50, vgg16_bn

from tqdm import tqdm

from al import ActivePool
from utils import Tracker
from evaluate import ece_loss, nll


def create_active_pool(config) -> ActivePool:
    
    if config.dataset_name == 'cifar10':

        mean = [0.4914, 0.4822, 0.4465]
        std  = [0.2470, 0.2435, 0.2616]
        root = os.path.join(config.dataset_path, config.dataset_name)

        train_transform = T.Compose([
            T.RandomHorizontalFlip(), 
            T.ToTensor(), 
            T.Normalize(mean, std)
        ])
        test_transform  = T.Compose([
            T.ToTensor(), 
            T.Normalize(mean, std)
        ])

        train_set = CIFAR10(root=root, train=True,  download=True, transform=train_transform)
        query_set = CIFAR10(root=root, train=True,  download=True, transform=test_transform)
        test_set  = CIFAR10(root=root, train=False, download=True, transform=test_transform)

    elif config.dataset_name == 'cifar100':

        mean = [0.5071, 0.4865, 0.4409]
        std  = [0.2673, 0.2564, 0.2762]
        root = os.path.join(config.dataset_path, config.dataset_name)

        train_transform = T.Compose([
            T.RandomHorizontalFlip(), 
            T.ToTensor(), 
            T.Normalize(mean, std)
        ])
        test_transform  = T.Compose([
            T.ToTensor(), 
            T.Normalize(mean, std)
        ])

        train_set = CIFAR100(root=root, train=True,  download=True, transform=train_transform)
        query_set = CIFAR100(root=root, train=True,  download=True, transform=test_transform)
        test_set  = CIFAR100(root=root, train=False, download=True, transform=test_transform)

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
        query_set = ImageFolder(train_dir, transform=test_transform)
        test_set  = ImageFolder(val_dir,   transform=test_transform)


    else:
        raise NotImplementedError

    pool = ActivePool(train_set=train_set, query_set=query_set, test_set=test_set, batch_size=config.batch_size)
            
    return pool


def init_model(config) -> nn.Module:

    if config.dataset_name == 'cifar10':
        num_classes = 10
    elif config.dataset_name == 'cifar100':
        num_classes = 100
    elif config.dataset_name == 'tiny':
        num_classes = 200
    else:
        raise NotImplementedError

    if config.arch == "resnet18":
        model = resnet18(pretrained=False, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()

    elif config.arch == "resnet50":
        model = resnet50(pretrained=False, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
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

    elif config.arch == "densenet121":
        raise NotImplementedError 

    else:
        raise ValueError
    
    return model


def init_optimizer(config, model: nn.Module) -> optim.Optimizer:
        
    if config.optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)
    elif config.optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise ValueError
    
    return optimizer


def init_model_and_optimizer(config) -> Tuple[nn.Module, optim.Optimizer]:
    model = init_model(config)
    optimizer = init_optimizer(config, model)
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
    for imgs, lbls in tqdm(dataloader, leave=False, desc='Train'):

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

def train_epoch_regul(model: nn.Module, swa_model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, lr_scheduler: Optional[LambdaLR]=None, device: Optional[torch.device]=None, regul_mode=False, lamb: float=1e-2) -> float:

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

        if regul_mode:
            swa_model.to(device)
            swa_reg = torch.tensor(0.).to(device)
            for p_swa, p_model in zip(swa_model.parameters(), model.parameters()):
                swa_reg += torch.norm(p_model - p_swa)
            loss += lamb * swa_reg

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
    top5_ = top_k_accuracy_score(targets_np, probs_np, k=5, labels=np.arange(probs_np.shape[1]))

    return {"acc": float(acc_), "nll": float(nll_), "ece": float(ece_), "top5": float(top5_)}


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
    for imgs, lbls in tqdm(dataloader, leave=False, desc='Predict'):

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
    ens_metrics = {f"ens/{k}": v for k, v in ens_metrics.items()}
    ens_metrics['ens/mean_acc'] = float(np.mean(accs))
    ens_metrics['ens/std_acc']  = float(np.std(accs))

    return ens_metrics