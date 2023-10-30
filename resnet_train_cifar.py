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

def train():
    train_loader, test_loader= load_cifar10(int(sys.argv[1]), 2)

    # Create the ResNet model and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if(sys.argv[2] == 'short'):
        model = ResNet(BasicBlock, [9,9,9]).to(device)
    else:
        model = ResNet(BasicBlockNoShort, [9,9,9]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
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

    if(sys.argv[2] == 'short'):
        torch.save(model, "weights/resmshortcut1.pt")
    else:
        torch.save(model, "weights/reslongcutm1.pt")
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