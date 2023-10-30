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
from visfuncs  import interpolate, move1D, move2D
from data import load_cifar10
import numpy as np
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--weight_path', type=str, default='resm1.pt', help='Path to the model')

args = parser.parse_args()


def give_model(path, device):
    model_init = torch.load(path, map_location=device)
    model_init.eval()

    return model_init

def plot_loss_acc_inter(model1, model2, criterion, dataloader, device):
    stepsize = 20
    loss, accs, x = interpolate(dataloader, criterion, model1, model2, stepsize, device, log=True)
    np.save("loss.npy", loss)
    np.save("accs.npy", accs)
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
    stepsize = 20
    loss, accs, x, y = move2D(dataloader, criterion, model, dirn1, dirn2, stepsize, stepsize, device, log=True)
    np.save('2d'+args.modelname+'.npy', np.array([[loss], [accs], [x], [y]]))
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, loss)
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, accs)
    plt.show()

train_loader, test_loader = load_cifar10(128, 2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model1 = give_model(args.weightpath, device)
# model2 = give_model('weights/resm2.pt', device)

# Uncomment for printing the parameters of the model

# for name, param in model2.named_parameters():
    # print(name)
    # print(param.shape)

criterion = nn.CrossEntropyLoss()

#plot_loss_acc_inter(model1, model2, criterion, train_loader, device)

# dirn = ResNet()
# for name, param in dirn.named_parameters():
#     param.data = torch.randn_like(param.data)

# plot_loss_acc_move1D(model1, dirn, criterion, test_loader, device)

dirn1 = model1.clone()
dirn1.to(device)
for param, m_param in zip(dirn1.parameters(), model1.parameters()):
     param.data = torch.randn_like(param.data)
     param.data = param.data / torch.linalg.norm(param.data)
     param.data *= m_param

dirn2 = model1.clone()
dirn2.to(device)
for param, m_param in zip(dirn2.parameters(), model1.parameters()):
    param.data = torch.randn_like(param.data)
    param.data = param.data / torch.linalg.norm(param.data)
    param.data *= m_param 

plot_loss_acc_move2D(model1, dirn1, dirn2, criterion, train_loader, device)
