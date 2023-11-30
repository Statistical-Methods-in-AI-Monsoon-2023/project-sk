# Outputs the eigenvalues of the hessian matrix at every point of the loss landscape

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
from models.resnet import ResNet, BasicBlock, BasicBlockNoShort
from visfuncs  import interpolate, move1D, move2D
from models import give_model, gen_unique_id
from data import load_cifar10
import numpy as np
import argparse
import torch.multiprocessing as mp
from scipy.sparse.linalg import LinearOperator, eigsh
import time

parser = argparse.ArgumentParser()

parser.add_argument('--weight_path1', type=str, help='Path to the weights file for the first model')
parser.add_argument('--weight_path2', type=str, help='Path to the weights file for the second model')
parser.add_argument('--model', type=str, help='Name of the model',required=True)
parser.add_argument('--filternorm', type=bool, default=False, help='Type of visualiztion to use')
parser.add_argument('--range', type=int, default=20, help='In [-1, 1] the number of steps to take in one direction(same for both x and y). Higher the number, higher the resolution of the plot will be')


def load_model_with_weights(path, device):
    model_init = torch.load(path, map_location=device)
    model_init, wd  = model_init['model'], model_init['weight_norm']
    # try:
    #     net = ResNet(BasicBlockNoShort, [9,9,9])
    #     net.load_state_dict(model_init['state_dict'])
    #     net.eval()
    #     return net
    # except:
    #     pass
    model_init.eval()

    return model_init, wd

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
    return loss/len(dataloader.dataset), acc


def vis(rank, model1, model2, dirn, criterion, steps, indices, output):

    model1.to(rank)
    model2.to(rank)
    dirn.to(rank)
    train_loader, _ = load_cifar10(128, 2)
    # print(vis_model.parameters().is_cuda())
    for s, step in enumerate(steps):
        for i, j, k in zip(model2.parameters(), model1.parameters(), dirn.parameters()):
            k.data = step*i.data + (1-step)*j.data
        loss, acc = give_loss_acc(train_loader, dirn, criterion, rank)
        print(f"GPU {rank} : Step {step}, Loss {loss}, Acc {acc}")
        output[indices[s], 0] = loss
        output[indices[s], 1] = acc


if __name__ == "__main__":
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nprocs = torch.cuda.device_count()
    workers = []
    torch.manual_seed(6)
    model1, wd1 = load_model_with_weights(args.weight_path1, 'cpu')
    model2, wd2 = load_model_with_weights(args.weight_path2, 'cpu')
    criterion = nn.CrossEntropyLoss()
    args.dataset = "cifar10"
    dirn = give_model(args)
    if(args.filternorm):
        for param, m_param in zip(dirn.parameters(), model1.parameters()):
            if(len(m_param.shape) == -1):
                param = m_param
                continue
            param.data = torch.randn_like(param.data)
            param.data = param.data / torch.linalg.norm(param.data)
            param.data *= m_param
    else:
        for param, m_param in zip(dirn.parameters(), model2.parameters()):
            if(len(m_param.shape) == -1):
                param = m_param.data
                continue
            param.data = torch.randn_like(param.data)

    alpha = torch.linspace(-1, 1.5, args.range)
    
    num_per_proc = alpha.shape[0] // nprocs
    steps = [alpha[i*num_per_proc:(i+1)*num_per_proc] for i in range(nprocs-1)]
    steps.append(alpha[(nprocs-1)*(num_per_proc):])
    indices = [torch.arange(i*num_per_proc, (i+1)*num_per_proc) for i in range(nprocs-1)]
    indices.append(torch.arange((nprocs-1)*num_per_proc, alpha.shape[0]))
    print(steps)
    print(indices)
    # Loss and Acc
    output = torch.zeros(alpha.shape[0], 2)
    output.share_memory_()

    mp.set_start_method('spawn', force=True)

    for i in range(nprocs):
        p = mp.Process(target=vis, args=[i, model1, model2, dirn, criterion, steps[i], indices[i], output])
        p.start()
        workers.append(p)

    for i in range(nprocs):
        workers[i].join()

    output = output.numpy()
    np.savez('2dsave.npz', loss=output[:,0], acc=output[:,1], x=alpha.numpy(), wd1 = wd1, wd2 = wd2)
    #np.save("2dsave.npy", np.array([[output[..., 0]], [output[..., 1]], alpha.numpy()]))
