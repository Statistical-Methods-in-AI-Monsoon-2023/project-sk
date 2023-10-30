import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
from models.cnn import CNN
from visfuncs  import interpolate, move1D, move2D
import numpy as np

def give_model(path):
    model_init = CNN()
    model_init = torch.load(path, map_location=torch.device('cpu'))
    model_init.eval()

    return model_init

def plot_loss_acc_inter(model1, model2, criterion, dataloader, device):
    stepsize = 20
    loss, accs, x = interpolate(dataloader, criterion, model1, model2, stepsize, device, log=True)
    plt.plot(x, loss)
    plt.show()
    plt.plot(x, accs)
    plt.show()

def plot_loss_acc_move1D(model, dirn, criterion, dataloader, device):
    stepsize = 20
    loss, accs, x = move1D(dataloader, criterion, model, dirn, stepsize, device, log=True)
    plt.plot(x, loss)
    plt.show()
    plt.plot(x, accs)
    plt.show()

def plot_loss_acc_move2D(model, dirn1, dirn2, criterion, dataloader, device):
    stepsize = 10
    loss, accs, x, y = move2D(dataloader, criterion, model, dirn1, dirn2, stepsize, stepsize, device, log=True)
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, loss)
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, accs)
    plt.show()

# Define data transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Load and preprocess the MNIST dataset
test_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32)

model1 = give_model('weights/m1.pt')
model2 = give_model('weights/m2.pt')

# Uncomment for printing the parameters of the model

# for name, param in model2.named_parameters():
    # print(name)
    # print(param.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# plot_loss_acc_inter(model1, model2, criterion, test_loader, device)

# dirn = CNN()
# for name, param in dirn.named_parameters():
#     param.data = torch.randn_like(param.data)

# plot_loss_acc_move1D(model1, dirn, criterion, test_loader, device)

dirn1 = CNN()
for param, m_param in zip(dirn1.parameters(), model1.parameters()):
    param.data = torch.randn_like(param.data)
    param.data = param.data / torch.linalg.norm(param.data)
    param.data *= m_param

dirn2 = CNN()
for param, m_param in zip(dirn2.parameters(), model2.parameters()):
    param.data = torch.randn_like(param.data)
    param.data = param.data / torch.linalg.norm(param.data)
    param.data *= m_param 

plot_loss_acc_move2D(model1, dirn1, dirn2, criterion, test_loader, device)