import os
import json
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from .dataset_lmdb import COOPLMDBDataset
from .abide import ABIDE
from .const import GTSRB_LABEL_MAP, IMAGENETNORMALIZE

def prepare_dataset(dataset, batch_size=128, pin_memory=False):
    from cfg import data_path
    data_path = os.path.join(data_path, dataset)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    if dataset == "mnist":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = train_transform
        train_data = datasets.MNIST(root = data_path, train = True, download = True, transform = train_transform)
        train_loader = DataLoader(train_data, batch_size, shuffle = True, num_workers=0)
        test_data = datasets.MNIST(root = data_path, train = False, download = True, transform = test_transform)
        test_loader = DataLoader(test_data, batch_size, shuffle = False, num_workers=0)
        cls_num = 10
    elif dataset == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_data = datasets.CIFAR10(root = data_path, train = True, download = True, transform = train_transform)
        train_loader = DataLoader(train_data, batch_size, shuffle = True, num_workers=0, pin_memory=pin_memory)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = True, transform = test_transform)
        test_loader = DataLoader(test_data, batch_size, shuffle = False, num_workers=0, pin_memory=pin_memory)
        cls_num = 10
    else:
        raise NotImplementedError
    return {
        'train': train_loader,
        'test': test_loader,
    }, cls_num
def refine_classnames(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names


def get_class_names_from_split(root):
    with open(os.path.join(root, "split.json")) as f:
        split = json.load(f)["test"]
    idx_to_class = OrderedDict(sorted({s[-2]: s[-1] for s in split}.items()))
    return list(idx_to_class.values())


def prepare_expansive_data(dataset, data_path):
    print(data_path)
    data_path = os.path.join(data_path, dataset)
    print(data_path)
    if dataset == "cifar10":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.CIFAR10(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "cifar100":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.CIFAR100(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "gtsrb":
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(list(GTSRB_LABEL_MAP.values())),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "svhn":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.SVHN(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': [f'{i}' for i in range(10)],
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "abide":
        preprocess = transforms.ToTensor()
        D = ABIDE(root = data_path)
        X_train, X_test, y_train, y_test = train_test_split(D.data, D.targets, test_size=0.1, stratify=D.targets, random_state=1)
        train_data = ABIDE(root = data_path, transform = preprocess)
        train_data.data = X_train
        train_data.targets = y_train
        test_data = ABIDE(root = data_path, transform = preprocess)
        test_data.data = X_test
        test_data.targets = y_test
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': ["non ASD", "ASD"],
            'mask': D.get_mask(),
        }
    elif dataset in ["food101", "eurosat", "sun397", "ucf101", "stanfordcars", "flowers102"]:
        preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=8),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((128, 128)),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        
        preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        print(2)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)

        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=1),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=1),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((128, 128)),
        }
    else:
        raise NotImplementedError(f"{dataset} not supported")
    return loaders, configs


def prepare_additive_data(dataset, data_path, preprocess):
    data_path = os.path.join(data_path, dataset)
    if dataset == "cifar10":
        train_data = datasets.CIFAR10(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = False, transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
    elif dataset == "cifar100":
        train_data = datasets.CIFAR100(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = False, transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
    elif dataset == "svhn":
        train_data = datasets.SVHN(root = data_path, split="train", download = False, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = False, transform = preprocess)
        class_names = [f'{i}' for i in range(10)]
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
    elif dataset in ["food101", "sun397", "eurosat", "ucf101", "stanfordcars", "flowers102"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=8),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=8),
        }
    elif dataset == "gtsrb":
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
    elif dataset == "abide":         
        D = ABIDE(root = data_path)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
        ])
        X_train, X_test, y_train, y_test = train_test_split(D.data, D.targets, test_size=0.1, stratify=D.targets, random_state=1)
        train_data = ABIDE(root = data_path, transform = preprocess)
        train_data.data = X_train
        train_data.targets = y_train
        test_data = ABIDE(root = data_path, transform = preprocess)
        test_data.data = X_test
        test_data.targets = y_test
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=2),
        }
        class_names = ["non ASD", "ASD"]
    else:
        raise NotImplementedError(f"{dataset} not supported")

    return loaders, class_names


def prepare_gtsrb_fraction_data(data_path, fraction, preprocess=None):
    data_path = os.path.join(data_path, "gtsrb")
    assert 0 < fraction <= 1
    new_length = int(fraction*26640)
    indices = torch.randperm(26640)[:new_length]
    sampler = SubsetRandomSampler(indices)
    if preprocess == None:
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, sampler=sampler, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(list(GTSRB_LABEL_MAP.values())),
            'mask': np.zeros((32, 32)),
        }
        return loaders, configs
    else:
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        loaders = {
            'train': DataLoader(train_data, 128, sampler=sampler, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        return loaders, class_names