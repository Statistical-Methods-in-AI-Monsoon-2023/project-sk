from sklearn.decomposition import PCA
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
from data import load_cifar10
import numpy as np
import argparse
import torch.multiprocessing as mp
import os
from models import give_model


parser = argparse.ArgumentParser(description='Plot loss and accuracy')
parser.add_argument('--folderpath', type=str)
parser.add_argument('--show', action='store_true')
parser.add_argument('--json', action='store_true')
parser.add_argument('--model', type=str, help='Name of the model',required=True)

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


def vis(rank, model, dirn1, dirn2, criterion, steps, indices, output):

    model.to(rank)
    vis_model = copy.deepcopy(model)
    dirn1.to(rank)
    dirn2.to(rank)
    train_loader, _ = load_cifar10(128, 2)
    # print(vis_model.parameters().is_cuda())
    for s, step in enumerate(steps):
        # idx is [a, b]
        a, b = step
        for i, d1, d2, k in zip(model.parameters(), dirn1.parameters(), dirn2.parameters(), vis_model.parameters()):
                k.data = i.data + a*d1.data + b*d2.data
        loss, acc = give_loss_acc(train_loader, vis_model, criterion, rank)
        print(f"GPU {rank} : Loss {loss}, Acc {acc}")
        output[indices[s], 0] = loss
        output[indices[s], 1] = acc


def plot_pca_traj(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nprocs = torch.cuda.device_count()
    workers = []
    criterion = nn.CrossEntropyLoss()

    models = []
    for file in os.listdir(args.folderpath):
        if file.endswith(".pt"):
            model = torch.load(os.path.join(args.folderpath, file), map_location=device)
            model.eval()
            models.append(model)
    concats = []
    for i in range(len(models)):
        model = models[i]
        concat = [param.reshape(-1) for param in model.parameters()]
        concat = torch.cat(concat).cpu().detach().numpy()
        concats.append(concat)

    diff_concats = []

    for i in range(len(concats) - 1):
        diff_concats.append(concats[i] - concats[-1])
    
    diff_concats = np.array(diff_concats)
    print(diff_concats.shape)
    pca = PCA(n_components=2)
    pca.fit(diff_concats)
    eigvecs = pca.components_

    # print(eigvecs)

    diff_concats = torch.tensor(diff_concats).to(device)
    eigvecs = torch.tensor(eigvecs).to(device)
    eigvecs = eigvecs / torch.linalg.norm(eigvecs, dim=1).reshape(-1, 1)
    x_vals, y_vals = eigvecs @ diff_concats.T
    x_vals = x_vals.detach().cpu().numpy()
    y_vals = y_vals.detach().cpu().numpy()
    # xlist, ylist = [], []
    # for i in range(len(models) -1 ):
    #     xlist.append((torch.dot(eigvecs[0], diff_concats[i])/torch.norm(eigvecs[0])).detach().cpu().item())
    #     ylist.append((torch.dot(eigvecs[1], diff_concats[i])/torch.norm(eigvecs[1])).detach().cpu().item())

    # plot trajectory

    print(x_vals.shape, y_vals.shape)
    print(x_vals, y_vals)

    plt.figure()
    plt.scatter(x_vals, y_vals)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Trajectory of model parameters")
    
    if args.show:
        plt.show()
    else:
        plt.savefig(f"results/{args.model}-pca.png")

    # if "noshort" in args.weights:
    #     block = BasicBlockNoShort
    # else:
    #     block = BasicBlock

    # dirn1 = ResNet(block, [9,9,9])
    # # dirn1.to(device)
    # for param, m_param in zip(dirn1.parameters(), model.parameters()):
    #     if(len(m_param.shape) == 1):
    #         param = m_param
    #         continue
    #     param.data = torch.randn_like(param.data)
    #     param.data = param.data / torch.linalg.norm(param.data)
    #     param.data *= m_param

    # dirn2 = ResNet(block, [9,9,9])
    # # dirn2.to(device)
    # for param, m_param in zip(dirn2.parameters(), model.parameters()):
    #     if(len(m_param.shape) == 1):
    #         param = m_param
    #         continue
    #     param.data = torch.randn_like(param.data)
    #     param.data = param.data / torch.linalg.norm(param.data)
    #     param.data *= m_param 

    # alpha = torch.linspace(-1, 1, 20)
    # beta = torch.linspace(-1, 1, 20)
    # mesh_x, mesh_y = torch.meshgrid(alpha, beta)
    # mesh = torch.cat([mesh_x.unsqueeze(0), mesh_y.unsqueeze(0)], 0).permute(1, 2, 0).reshape(-1, 2)
    
    # num_per_proc = mesh.shape[0] // nprocs
    # steps = [mesh[i*num_per_proc:(i+1)*num_per_proc, :] for i in range(nprocs-1)]
    # steps.append(mesh[(nprocs-1)*(num_per_proc):, :])
    # indices = [torch.arange(i*num_per_proc, (i+1)*num_per_proc) for i in range(nprocs-1)]
    # indices.append(torch.arange((nprocs-1)*num_per_proc, mesh.shape[0]))

    # # Loss and Accuracy Values
    # output = torch.zeros(mesh.shape[0], 2)
    # output.share_memory_()

    # mp.set_start_method('spawn', force=True)

    # for i in range(nprocs):
    #     p = mp.Process(target=vis, args=[i, model, dirn1, dirn2, criterion, steps[i], indices[i], output])
    #     p.start()
    #     workers.append(p)

    # for i in range(nprocs):
    #     workers[i].join()

    # loss = output[:, 0].reshape(mesh_x.shape)
    # acc = output[:, 1].reshape(mesh_x.shape)

    # plt.figure()
    # plt.contourf(mesh_x, mesh_y, loss, 20)
    # plt.colorbar()
    # plt.xlabel("alpha")
    # plt.ylabel("beta")
    # plt.title("Loss Contour")
    # plt.savefig("loss_contour.png")
            

if __name__ == "__main__":
    args = parser.parse_args()
    plot_pca_traj(args)