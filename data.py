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

class XOR(Dataset):
    def __init__(self):
        self.x = torch.tensor([[0, 0],
                               [1, 0],
                               [0, 1],
                               [1, 1]]).to(torch.float32)
        self.y = torch.tensor([0, 1, 1, 0])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class IDD(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.file_list)

    def processAnnots(self, annot):
        num_anns = len(annot)
        data = torch.empty((num_anns, 6))
        for i, ann in enumerate(annot):
            vals = ann[:-1].split()
            data[i, 0] = int(vals[0])
            data[i, 1] = float(vals[1])
            data[i, 2] = float(vals[2])
            data[i, 3] = float(vals[3])
            data[i, 4] = float(vals[4])
            data[i, 5] = int(vals[5])
        return data

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_list[idx])
        annot_file = os.path.join(self.root_dir, self.file_list[idx].split('.')[0]) + ".txt"
        image = Image.open(img_name).convert('RGB')


        f = open(annot_file)
        annots = f.readlines()
        downsampled_size = [20, 20]
        
        if self.transform:
            image = self.transform(image) / 255

        if(len(annots) == 0):
            return image, torch.zeros(downsampled_size)
        
        return image, torch.zeros(downsampled_size)
        data = self.processAnnots(annots)


        # data is in (cls, xc, yc, w, h, id) format
        # convert to a heatmap of downsampled size and logit probs 
        centers = data[:, 1:3]
        pixels = centers * torch.tensor(downsampled_size)
        pixels = pixels.to(torch.long).T
        target_hm = torch.zeros(downsampled_size)#, dtype=torch.long)
        target_hm[pixels[:, 0], pixels[:, 1]] = 1
        return image, target_hm
    
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


def load_mnist(batch_size, num_workers, distributed=False):
    # Define data transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Load and preprocess the MNIST dataset
    trainset = MNIST(root='./data', train=True, transform=transform, download=True)
    testset = MNIST(root='./data', train=False, transform=transform)
    
    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    
    train_sampler = None
    test_sampler = None

    if(distributed):
        train_sampler = DistributedSampler(trainset)
        test_sampler = DistributedSampler(testset)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, **kwargs, sampler=train_sampler)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, sampler=test_sampler)

    return train_loader, test_loader

def load_cifar10(batch_size, num_workers, distributed=False, augment=True):

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        ])
    
    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    #testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if(augment):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    
    train_sampler = None
    test_sampler = None

    if(distributed):
        train_sampler = DistributedSampler(trainset)
        test_sampler = DistributedSampler(testset)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, **kwargs, sampler=train_sampler)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, sampler=test_sampler)

    return train_loader, test_loader

def load_xor(batch_size, num_workers, distributed=False):
    
    trainset = XOR()
    testset = XOR()

    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    
    train_sampler = None
    test_sampler = None

    if(distributed):
        train_sampler = DistributedSampler(trainset)
        test_sampler = DistributedSampler(testset)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, **kwargs, sampler=train_sampler)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, sampler=test_sampler)

    return train_loader, test_loader

def load_dataset(args, distributed=True):
    if(args.dataset == 'cifar10'):
        return load_cifar10(args.batch_size, 4, distributed=distributed, augment=True)
    elif(args.dataset == "mnist"):
        return load_mnist(args.batch_size, 4, distributed=distributed)
    elif(args.dataset == "xor"):
        return load_xor(args.batch_size, 4, distributed=distributed)
