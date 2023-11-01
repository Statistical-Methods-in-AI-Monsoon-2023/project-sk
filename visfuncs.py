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
from models.resnet import ResNet, BasicBlock

@torch.no_grad()
def give_loss_acc(dataloader, model, criterion, device):
    '''
    dataloader is a torch.Dataloader instance
    criterion is the loss function
    model is the model architecture with its parameters
    device is the device on which this code is running on

    returns the loss and accuracy 
    '''
    loss = 0
    num = 0
    corr = 0

    for inputs, labels in dataloader:
        num += inputs.shape[0]
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss += criterion(outputs, labels).item()
        corr += torch.where(labels == torch.argmax(outputs, -1))[0].shape[0]

    acc = corr / num
    return loss, acc


def interpolate(dataloader, criterion, model1, model2, stepsize, device, log=False):
    '''
    dataloader is a torch.Dataloader instance
    criterion is the loss function
    model1 is the model architecture with its parameters
    model2 is the model architecture with its parameters
    device is the device on which this code is running on
    log is whether to print the loss and acc at each step

    returns the loss and accuracy lists
    '''
    alpha = torch.linspace(0, 1, stepsize)
    #model = copy.deepcopy(model2)
    model = ResNet(BasicBlock, [9,9,9]).to(device)
    losses = []
    accs = []
    for a in alpha:
        for i, j, k in zip(model1.parameters(), model2.parameters(), model.parameters()):
            k.data = (i.data * a + j.data * (1 - a))

        loss, acc = give_loss_acc(dataloader, model, criterion, device)
        if(log):
            print(f"Loss: {loss}, Acc: {acc}, Alpha: {a}")
        losses.append(loss)
        accs.append(acc)
    
    return torch.tensor(losses).numpy(), torch.tensor(accs).numpy(), alpha.numpy()

def move1D(dataloader, criterion, model1, dirn, stepsize, device, log=False):
    '''
    dataloader is a torch.Dataloader instance
    criterion is the loss function
    model1 is the model architecture with its parameters
    dirn is the same model architecture with weights as the dirn vectors for the corresponding weight
    device is the device on which this code is running on
    log is whether to print the loss and acc at each step

    returns the loss and accuracy lists
    '''

    alpha = torch.linspace(-1, 1, stepsize)
    model = copy.deepcopy(model1)
    losses = []
    accs = []
    for a in alpha:
        for i, j, k in zip(model1.parameters(), dirn.parameters(), model.parameters()):
            k.data = i.data + a*j.data

        loss, acc = give_loss_acc(dataloader, model, criterion, device)
        if(log):
            print(loss)
            print(acc)
        losses.append(loss)
        accs.append(acc)
    
    return torch.tensor(losses).numpy(), torch.tensor(accs).numpy(), alpha.numpy()

def move2D(dataloader, criterion, model1, dirn1, dirn2, stepsize1, stepsize2, device, log=False):
    '''
    dataloader is a torch.Dataloader instance
    criterion is the loss function
    model1 is the model architecture with its parameters
    dirn1 is the same model architecture with weights as the dirn vectors for the corresponding weight
    dirn2 is the same model architecture with weights as the dirn vectors for the corresponding weight
    stepsize1 is the number of steps to be taken along x
    stepsize2 is the number of steps to be taken along y
    device is the device on which this code is running on
    log is whether to print the loss and acc at each step

    returns the loss and accuracy lists
    '''

    alpha = torch.linspace(-100, 100, stepsize1)
    beta = torch.linspace(-100, 100, stepsize2)

    model = copy.deepcopy(model1)

    losses = []
    accs = []

    for a in alpha:
        for b in beta:
            for i, d1, d2, k in zip(model1.parameters(), dirn1.parameters(), dirn2.parameters(), model.parameters()):
                k.data = i.data + a*d1.data + b*d2.data

            loss, acc = give_loss_acc(dataloader, model, criterion, device)
            if(log):
                print(loss)
                print(acc)
            losses.append(loss)
            accs.append(acc)
    
    losses = torch.tensor(losses).reshape(stepsize1, stepsize2).numpy()
    accs = torch.tensor(accs).reshape(stepsize1, stepsize2).numpy()
    x, y = torch.meshgrid(alpha, beta)
    return losses, accs, x.numpy(), y.numpy()
