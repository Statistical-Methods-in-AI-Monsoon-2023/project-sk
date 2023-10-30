import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.vgg import VGG, vgg_cfgs
from data import load_cifar10
import sys


import argparse

parser = argparse.ArgumentParser(description='Train a VGG model on CIFAR10')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_layers', type=int, default=11, help='Number of layers in the VGG model')
args = parser.parse_args()


def train():
    train_loader, test_loader= load_cifar10(int(args.batch_size), 2)

    # Create the ResNet model and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vgg_num_layers = str(args.num_layers)
    vgg_id = "vgg" + vgg_num_layers
    if vgg_id in vgg_cfgs:
        model = VGG(vgg_cfgs[vgg_id], True).to(device)
    else:
        print("Invalid VGG Configuration")
        return

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

    torch.save(model, f"weights/{vgg_id}.pt")
    # Test the model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy}%')

if __name__ == '__main__':
    train()