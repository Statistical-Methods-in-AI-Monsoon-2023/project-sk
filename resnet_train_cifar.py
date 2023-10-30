import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.resnet import ResNet, BasicBlock, BasicBlockNoShort
from data import load_cifar10
import sys

import argparse

parser = argparse.ArgumentParser(description='Train a ResNet model on CIFAR10')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--short', action='store_true', help='Use the short version of ResNet')
args = parser.parse_args()

def train():
    train_loader, test_loader= load_cifar10(int(args.batch_size),2)

    # Create the ResNet model and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if(args.short):
        model = ResNet(BasicBlock, [9,9,9]).to(device)
    else:
        model = ResNet(BasicBlockNoShort, [9,9,9]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model_unique_id = f"resnet-{
        'short' if args.short else 'noshort'
    }-{args.batch_size}-{args.epochs}epochs"

    # Training loop
    num_epochs = args.epochs
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

    if(args.short):
        torch.save(model, f"weights/{model_unique_id}.pt")
    else:
        torch.save(model, f"weights/{model_unique_id}.pt")
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