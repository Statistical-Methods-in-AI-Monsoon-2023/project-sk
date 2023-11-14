import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset

import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class IDD(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_list[idx])
        annot_file = os.path.join(self.root_dir, self.file_list[idx].split('.')[0]) + ".txt"
        image = Image.open(img_name).convert('RGB')

        f = open(annot_file)
        annots = f.readlines()

        if self.transform:
            image = self.transform(image)

        return image
    
def load_idd(data_root, batch_size, num_workers, distributed=False):

    # Define the transformations to be applied to the images
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Adjust the size as needed
        transforms.ToTensor(),
    ])

    # Create an instance of the CustomDataset
    trainset = IDD(root_dir=data_root, transform=transform)    
    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    train_sampler = None
    test_sampler = None

    if(distributed):
        train_sampler = DistributedSampler(trainset)

    # Create a DataLoader for batching and shuffling
    data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, **kwargs, sampler=train_sampler)
    
    return data_loader, None


def load_mnist(batch_size):
    # Define data transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Load and preprocess the MNIST dataset
    train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader

def load_cifar10(batch_size, num_workers, distributed=False):

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    
    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    
    train_sampler = None
    test_sampler = None

    if(distributed):
        train_sampler = DistributedSampler(trainset)
        test_sampler = DistributedSampler(testset)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, **kwargs, sampler=train_sampler)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, sampler=test_sampler)

    return train_loader, test_loader
