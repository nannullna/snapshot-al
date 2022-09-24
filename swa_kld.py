import os
from datetime import datetime
from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.swa_utils import SWALR

import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

from scripts.commons import init_model_and_optimizer, train_epoch, predict
from tqdm import tqdm, trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

DATASET_PATH = "/opt/datasets/cifar10"

def prepare_datsets(size: int):

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

    raw_datasets = CIFAR10(root=DATASET_PATH, train=True, download=True, transform=train_transform)

    train_set = Subset(raw_datasets, indices=list(range(size)))
    test_set  = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform)

    return {
        'train': train_set,
        'test': test_set
    }

@dataclass
class TrainingArguments:
    arch: str = 'resnet18'
    optimizer_type: str = 'sgd'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    dataset_name: str = 'cifar10'
    momentum: float = 0.9

SAVE_PATH = f'empirical/swa_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
NUM_ENSEMBLES = 5
BATCH_SIZE = 64
NUM_EPOCHS = 200
SWA_EPOCHS = 50
TRAIN_SIZES = [20000, 25000, 30000, 40000]

save_interval = SWA_EPOCHS // NUM_ENSEMBLES
save_at = [NUM_EPOCHS + (i+1)*save_interval for i in range(NUM_ENSEMBLES)]

config = TrainingArguments()
os.makedirs(SAVE_PATH)
print(f"Save path created at {SAVE_PATH}")
print(f"Checkpoint will be saved at {save_at} epochs.")

for train_size in TRAIN_SIZES:

    if train_size == 0:
        continue

    print(f"Start train size {train_size}.")

    datasets = prepare_datsets(train_size)
    train_loader = DataLoader(datasets['train'], batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(datasets['test'],  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model, optimizer = init_model_and_optimizer(config)
    scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate*10, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader))
    swa_scheduler = SWALR(optimizer, swa_lr=config.learning_rate*10, anneal_epochs=10)
    model.to(device)

    TOTAL_EPOCHS = NUM_EPOCHS + SWA_EPOCHS
    for epoch in trange(1, TOTAL_EPOCHS+1):
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler=scheduler if epoch <= NUM_EPOCHS else swa_scheduler, device=device)

        if epoch == NUM_EPOCHS:
            outputs = predict(model, test_loader, device=device)
            test_acc = (outputs['preds'] == outputs['targets']).float().mean().item()
            print(f"Test accuracy {test_acc:.3f}")

        if epoch in save_at:

            outputs = predict(model, test_loader, device=device)
            test_acc = (outputs['preds'] == outputs['targets']).float().mean().item()
            print(f"Test accuracy {test_acc:.3f}")
            
            logits_file = f"logits_size={train_size}_epoch={epoch}.pt"
            torch.save(outputs['logits'], os.path.join(SAVE_PATH, logits_file))
            print(f"Logits saved at {logits_file}.")

            model_file = f"model_size={train_size}_epoch={epoch}.pt"
            torch.save({"state_dict": model.state_dict()}, os.path.join(SAVE_PATH, model_file))
            print(f"Model saved at {model_file}")

