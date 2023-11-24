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
from models import give_model, gen_unique_id
from visfuncs  import interpolate, move1D, move2D
from data import load_cifar10, load_mnist
import numpy as np
import argparse
import torch.multiprocessing as mp
import os

parser = argparse.ArgumentParser()

parser.add_argument('--weight_path', type=str, help='Path to the weights file')
parser.add_argument('--model', type=str, help='Name of the model',required=True)
parser.add_argument('--range', type=int, default=20, help='In [-1, 1] the number of steps to take in one direction(same for both x and y). Higher the number, higher the resolution of the plot will be')
parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset to be used')

def load_model_with_weights(path, device):
    model_init = torch.load(path, map_location=device)
    model_init = model_init['model']
    # try:
    #     net = ResNet(BasicBlockNoShort, [9,9,9])
    #     net.load_state_dict(model_init['state_dict'])
    #     net.eval()
    #     return net
    # except:
    #     pass
    model_init.eval()

    return model_init

def load_dataset(args):
    if(args.dataset == 'cifar10'):
        return load_cifar10(256, 2, distributed=True)
    elif(args.dataset == "mnist"):
        return load_mnist(256, 2, distributed=True)

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


def vis(rank, model, dirn1, dirn2, criterion, steps, indices, output, args):

    model.to(rank)
    vis_model = copy.deepcopy(model)
    dirn1.to(rank)
    dirn2.to(rank)
    train_loader, _ = load_dataset(args)

    # print(vis_model.parameters().is_cuda())
    for s, step in enumerate(steps):
        # idx is [a, b]
        a, b = step
        for i, d1, d2, k in zip(model.parameters(), dirn1.parameters(), dirn2.parameters(), vis_model.parameters()):
                k.data = i.data + a*d1.data + b*d2.data

        loss, acc = give_loss_acc(train_loader, vis_model, criterion, rank)
        print(f"GPU {rank}: Step Horz {a}, Vert {b} Loss {loss}, Acc {acc}")
        output[indices[s], 0] = loss
        output[indices[s], 1] = acc


if __name__ == "__main__":
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nprocs = torch.cuda.device_count()
    workers = []

    model = load_model_with_weights(args.weight_path, 'cpu')
    criterion = nn.CrossEntropyLoss()

    dirn1 = give_model(args)
    # dirn1.to(device)
    for param, m_param in zip(dirn1.parameters(), model.parameters()):
        if(len(m_param.shape) == 1):
            param = m_param
            continue
        param.data = torch.randn_like(param.data)
        param.data = param.data / torch.linalg.norm(param.data)
        param.data *= torch.linalg.norm(m_param)

    dirn2 = give_model(args)
    # dirn2.to(device)
    for param, m_param in zip(dirn2.parameters(), model.parameters()):
        if(len(m_param.shape) == 1):
            param = m_param
            continue
        param.data = torch.randn_like(param.data)
        param.data = param.data / torch.linalg.norm(param.data)
        param.data *= torch.linalg.norm(m_param)

    alpha = torch.linspace(-1, 1, args.range)
    beta = torch.linspace(-1, 1, args.range)
    mesh_x, mesh_y = torch.meshgrid(alpha, beta)
    mesh = torch.cat([mesh_x.unsqueeze(0), mesh_y.unsqueeze(0)], 0).permute(1, 2, 0).reshape(-1, 2)
    
    num_per_proc = mesh.shape[0] // nprocs
    steps = [mesh[i*num_per_proc:(i+1)*num_per_proc, :] for i in range(nprocs-1)]
    steps.append(mesh[(nprocs-1)*(num_per_proc):, :])
    indices = [torch.arange(i*num_per_proc, (i+1)*num_per_proc) for i in range(nprocs-1)]
    indices.append(torch.arange((nprocs-1)*num_per_proc, mesh.shape[0]))

    # Loss and Accuracy Values
    output = torch.zeros(mesh.shape[0], 2)
    output.share_memory_()

    mp.set_start_method('spawn', force=True)

    for i in range(nprocs):
        p = mp.Process(target=vis, args=[i, model, dirn1, dirn2, criterion, steps[i], indices[i], output, args])
        p.start()
        workers.append(p)

    for i in range(nprocs):
        workers[i].join()

    output = output.reshape((mesh_x.shape[0], mesh_y.shape[0], 2)).numpy()
    unique_filename = gen_unique_id(args)
    np.save(f"results/plot_npy/{unique_filename}.npy", np.array([[output[..., 0]], [output[..., 1]], [mesh_x.numpy()], [mesh_y.numpy()]]))
