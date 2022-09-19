# SWAL: Stochastic Weight Active Learning

Official Implementation of Stochastic Weight Active Learning

---

# 1. Introduction

---

# 2. How to Use

## 2-1. How to Run

Run experiments on CIFAR-10 dataset with ResNet-18. Specify an acquisition function with `--query_type` and acquisition size with `--query_size` argument.

```bash
python scripts/train_snapshot.py -f configs/cifar10_resnet18.json --query_type vr

python scripts/train_ensemble.py -f configs/cifar10_resnet18.json --query_type vr

python scripts/train_sameinit.py -f configs/cifar10_resnet18.json --query_type vr
```

Run experiments on CIFAR-100 dataset with ResNet-18. Specify an acquisition function with `--query_type` and acquisition size with `--query_size` argument.

```bash
python scripts/train_snapshot.py -f configs/cifar100_resnet18.json --query_type vr

python scripts/train_ensemble.py -f configs/cifar100_resnet18.json --query_type vr

python scripts/train_sameinit.py -f configs/cifar100_resnet18.json --query_type vr

Run experiments on Tiny-ImageNet-200 dataset with ResNet-50. Specify an acquisition function with `--query_type` and acquisition size with `--query_size` argument.

```bash
python scripts/train_snapshot.py -f configs/tiny_resnet50.json --query_type vr

python scripts/train_ensemble.py -f configs/tiny_resnet50.json --query_type vr

python scripts/train_sameinit.py -f configs/tiny_resnet50.json --query_type vr
```

### 2-1-1. Download Datasets

CIFAR10 and CIFAR100 dataset will be downloaded using torchvision and saved to a directory provided as `--dataset_path`.

However, for Tiny Imagenet, you will need to download it first, unzip it, and specify a unzipped directory as `--dataset_path`. 

Our code will automatically create a file structure for the validation set.

Please edit config files according to your local setting before running the scripts. 

### 2-1-2. Edit Configs

Configuration files are in `./cifar/configs` and `./tiny-imagenet/configs` folder. You can directly edit these files for different hyperparameters. 


---
