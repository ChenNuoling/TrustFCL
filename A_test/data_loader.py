import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class TaskGenerator:
    def __init__(self, args):
        self.args = args
        self.dataset_name = getattr(args, 'dataset', 'cifar10').lower()
        
        model_type = getattr(args, 'model', '').lower()
        transform_list = []
        
        if 'vit' in model_type:
            transform_list.append(transforms.Resize((224, 224)))
        
        transform_list.append(transforms.ToTensor())
        
        self.transform = transforms.Compose(transform_list)

        self._load_dataset()

    def _load_dataset(self):
        dataset_name = self.dataset_name
        
        print(f"Loading {dataset_name.upper()}...")
        
        if dataset_name == 'cifar100':
            self.train_set = datasets.CIFAR100(
                root=self.args.data_root,
                train=True,
                download=True,
                transform=self.transform
            )
            self.test_set = datasets.CIFAR100(
                root=self.args.data_root,
                train=False,
                download=True,
                transform=self.transform
            )
            self.num_classes = 100
            
        elif dataset_name == 'cifar10':
            self.train_set = datasets.CIFAR10(
                root=self.args.data_root,
                train=True,
                download=True,
                transform=self.transform
            )
            self.test_set = datasets.CIFAR10(
                root=self.args.data_root,
                train=False,
                download=True,
                transform=self.transform
            )
            self.num_classes = 10
            
        elif dataset_name == 'mnist':
            self.train_set = datasets.MNIST(
                root=self.args.data_root,
                train=True,
                download=True,
                transform=self.transform
            )
            self.test_set = datasets.MNIST(
                root=self.args.data_root,
                train=False,
                download=True,
                transform=self.transform
            )
            self.num_classes = 10
            
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def get_loader(self, mode='train', num_samples=None):
        if mode == 'train':
            dataset = self.train_set
            default_samples = getattr(self.args, 'train_samples', 2000)
        else:
            dataset = self.test_set
            default_samples = getattr(self.args, 'test_samples', 64)
        
        num_samples = num_samples if num_samples is not None else default_samples
        
        indices = np.random.permutation(len(dataset))[:num_samples]
        subset = Subset(dataset, indices)
        
        shuffle = (mode == 'train')
        loader = DataLoader(
            subset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers if hasattr(self.args, 'num_workers') else 2
        )
        
        return loader