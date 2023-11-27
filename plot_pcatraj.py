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
parser.add_argument('--weight_path', type=str, help="path to the weight folder")
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--show', action='store_true')
parser.add_argument('--json', action='store_true')
parser.add_argument('--model', type=str, help='Name of the model',required=True)
parser.add_argument('--include_bn', type=bool, help='To include the batch norm in the parameters',default=True)


finalmodel = None

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
    train_loader, _ = load_cifar10(128, 2)

    for s, step in enumerate(steps):
        # idx is [a, b]
        a, b = step
        for i, d1, d2, k in zip(model.parameters(), dirn1.parameters(), dirn2.parameters(), vis_model.parameters()):
                if not args.include_bn:
                    if(len(i.shape) == 1):
                        continue
                k.data = i.data + a*d1.data + b*d2.data
        loss, acc = give_loss_acc(train_loader, vis_model, criterion, rank)
        print(f"GPU {rank} : Loss {loss}, Acc {acc}")
        output[indices[s], 0] = loss
        output[indices[s], 1] = acc


def get_eigvecs(args):
    models = []
    for file in os.listdir(args.weight_path):
        if file.endswith(".pt"):
            model = torch.load(os.path.join(args.weight_path, file), map_location=device)
            model.eval()
            models.append(model)

    global finalmodel
    finalmodel = models[-1]
    concats = []
    
    for i in range(len(models)):
        model = models[i]
        concat = [param.reshape(-1) for param in model.parameters()]
        filter = []
        concat = torch.cat(concat).cpu().detach().numpy()
        if not args.include_bn:
            for p in model.parameters():
                fn = torch.zeros_like if len(p.shape) == 1 else torch.ones_like
                filter.append(fn(p).reshape(-1))
            filter_concat = torch.cat(filter).cpu().detach().numpy()
            concat = concat * filter

        concats.append(concat)

    diff_concats = []

    for i in range(len(concats) - 1):
        diff_concats.append(concats[i] - concats[-1])
    diff_concats.append(torch.zeros_like(diff_concats[0]))
    
    diff_concats = np.array(diff_concats)
    print(diff_concats.shape)
    pca = PCA(n_components=2)
    pca.fit(diff_concats[:-1, :])
    eigvecs = pca.components_
    print("Variances: ", pca.explained_variance_ratio_)
    diff_concats = torch.tensor(diff_concats).to(device)
    eigvecs = torch.tensor(eigvecs).to(device)
    eigvecs = eigvecs / torch.linalg.norm(eigvecs, dim=1).reshape(-1, 1)

    # appending zeros for the last difference (that is the difference of the final weights with itself)
    return diff_concats, eigvecs

def give_proj(diff_concats, eigvecs):
    x_vals, y_vals = eigvecs @ diff_concats.T
    x_vals = x_vals.detach().cpu().numpy()
    y_vals = y_vals.detach().cpu().numpy()
    return x_vals, y_vals

def plot_trajectory(x_vals,y_vals):

    print(x_vals.shape, y_vals.shape)
    print(x_vals, y_vals)

    plt.scatter(x_vals, y_vals)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Trajectory of model parameters")
    
    if args.show:
        plt.show()
    else:
        plt.savefig(f"results/{args.model}-pca.png")


if __name__ == "__main__":
    args = parser.parse_args()
    filename = os.path.basename(args.weight_path)
    npy_data = f"results/plot_npy/{filename}-pcatraj-{args.steps}.npz"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nprocs = torch.cuda.device_count()
    criterion = nn.CrossEntropyLoss()
    diff_concats, eigvecs = get_eigvecs(args)
    x_vals, y_vals = give_proj(diff_concats, eigvecs)
    plot_trajectory(x_vals, y_vals)

    # calculate step size
    x_min, x_max = np.min(x_vals), np.max(x_vals)
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    x_std = np.std(x_vals)
    y_std = np.std(y_vals)
    x_min -= x_std
    x_max += x_std
    y_min -= y_std
    y_max += y_std
    print("x_min, x_max, y_min, y_max: ", x_min, x_max, y_min, y_max)

    v_ten1 = eigvecs[0]
    v_ten2 = eigvecs[1]

    model = finalmodel

    dirn1 = give_model(args)
    total_len = 0
    for param, m_param in zip(dirn1.parameters(), model.parameters()):
        param_len = param.reshape(-1).shape[0]
        param.data = v_ten1[total_len:total_len+param_len].reshape(param.shape) #* torch.linalg.norm(m_param)
        total_len += param_len

    dirn2 = give_model(args)
    total_len = 0
    for param, m_param in zip(dirn2.parameters(), model.parameters()):
        param_len = param.reshape(-1).shape[0]
        param.data = v_ten2[total_len:total_len+param_len].reshape(param.shape) #* torch.linalg.norm(m_param)
        total_len += param_len

    alpha = torch.linspace(x_min, x_max, args.steps)
    beta = torch.linspace(y_min, y_max, args.steps)
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
    print(output.shape)
    mp.set_start_method('spawn', force=True)
    
    workers = []
    for i in range(nprocs):
        p = mp.Process(target=vis, args=[i, model, dirn1, dirn2, criterion, steps[i], indices[i], output, args])
        p.start()
        workers.append(p)

    for i in range(nprocs):
        workers[i].join()
    
    print(output.shape)
    print(mesh_x.shape)
    output = output.reshape((mesh_x.shape[0], mesh_y.shape[0], 2)).numpy()
    loss = output[..., 0]
    loss = np.nan_to_num(loss, posinf=2*np.nanmax(loss), neginf=0)
    acc = output[..., 1]
    mesh_x = mesh_x.numpy()
    mesh_y = mesh_y.numpy()
    np.savez(npy_data, loss=loss, acc=acc, mesh_x=mesh_x, mesh_y=mesh_y)
    
    # output = np.load(npy_data)
    # loss,acc,mesh_x,mesh_y = output
    # loss = output['loss']
    # acc = output['acc']
    # mesh_x = output['mesh_x']
    # mesh_y = output['mesh_y']

    # print(loss.shape, acc.shape, mesh_x.shape, mesh_y.shape)

    # contour plot

    plt.contour(mesh_x, mesh_y, loss, cmap='RdGy',
                levels=20, vmin=loss.min(), vmax=loss.max())
    plt.colorbar()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Loss Contour Plot")
    if args.show:
        plt.show()
    
    plt.savefig(f"results/{filename}-{args.model}-pca-contour.png")

