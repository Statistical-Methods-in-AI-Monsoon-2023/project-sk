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

@torch.no_grad()
def give_loss_acc(dataloader, model, criterion, device):
    loss = 0
    num = 0
    corr = 0

    for inputs, labels in dataloader:
        num += inputs.shape[0]
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss += criterion(outputs, labels)
        corr += torch.where(labels == torch.argmax(outputs, -1))[0].shape[0]

    acc = corr / num
    return loss, acc


def interpolate(dataloader, criterion, model1, model2, device, log=False):
    stepsize = 20
    alpha = torch.linspace(0, 1, stepsize)
    model = copy.deepcopy(model1)
    losses = []
    accs = []
    for a in alpha:
        for i, j, k in zip(model1.parameters(), model2.parameters(), model.parameters()):
            k.data = (i.data * a + j.data * (1 - a))

        loss, acc = give_loss_acc(dataloader, model, criterion, device)
        if(log):
            print(loss)
            print(acc)
        losses.append(loss)
        accs.append(acc)
    
    return losses, accs