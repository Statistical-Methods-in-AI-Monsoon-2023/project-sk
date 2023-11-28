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
from data import load_dataset
import numpy as np
import argparse
import torch.multiprocessing as mp
from scipy.sparse.linalg import LinearOperator, eigsh
import time

parser = argparse.ArgumentParser()

parser.add_argument('--weight_path', type=str, help='Path to the weights file')
parser.add_argument('--model', type=str, help='Name of the model',required=True)
parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset to be used')
parser.add_argument('--dirn_file', type=str, required=True, help='The directions to be used for plotting the heatmap of the ratio of the eigenvalues')
parser.add_argument('--range', type=int, default=20, help='In [-1, 1] the number of steps to take in one direction(same for both x and y). Higher the number, higher the resolution of the plot will be')


def load_model_with_weights(path, device):
    model_init = torch.load(path, map_location=device)
    model_init = model_init['model']
    #try:
    #net = ResNet(BasicBlockNoShort, [9,9,9])
    #    net.load_state_dict(model_init['state_dict'])
   #     net.eval()
   #     return net
   # except:
    #    pass
    model_init.eval()

    return model_init


def give_minmax_eigs(dataloader, model, criterion, device):
    '''
    dataloader is a torch.Dataloader instance
    criterion is the loss function
    model is the model architecture with its parameters
    device is the device on which this code is running on

    returns the min and max eigenvalues
    '''

    model.num_calls = 0

    def hess_transform(v):
        # Computes H @ v, (v is an arbitrary vector)
        # v is a numpy array, first convert to tensor of the same shape as the model parameters
        model.num_calls += 1
        print(f"Call {model.num_calls}")
        v_tensor = torch.from_numpy(v).to(device)
        v_ten = []
        total_len = 0
        t0 = time.time()
        for param in model.parameters():
            param_len = param.reshape(-1).shape[0]
            v_ten.append(v_tensor[total_len:total_len+param_len].reshape(param.shape))
            total_len += param_len
        # v_ten is the same format as the model.params() list

        model.eval()
        model.zero_grad()

        # params = [param for param in model.parameters()]
        prob = 0.9

        for inputs, labels in dataloader:
            if(np.random.random(1) > prob):
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            loss = criterion(out, labels)
            t0 = time.time()
            # Create graph is used to calculate higher order gradients
            grads = torch.autograd.grad(outputs = loss, inputs = model.parameters(), create_graph=True)
            # print("Loss 1", time.time() - t0)
            t0 = time.time()
            # Simluate the dot product of the gradient with the v_ten elements
            accum_transform = torch.zeros(1, requires_grad=True).to(device)
            for param_grad, param_vec in zip(grads, v_ten):
                accum_transform += torch.sum(param_grad*param_vec)
            # print("Loop 2", time.time() - t0)
            
            # Keep on adding the gradients for the full dataset (The gradients will be accumulated into .grad)
            accum_transform.backward()
        
        final_transform = [param.grad.reshape(-1) for param in model.parameters()]
        print(f"GPU {device} Done")
        
        return torch.cat(final_transform).cpu().numpy()
    
    param_len = sum(param.reshape(-1).shape[0] for param in model.parameters())
    t0 = time.time()
    H = LinearOperator((param_len, param_len), matvec=hess_transform)
    print(f"GPU {device} Time Taken : {time.time() - t0}")
    maxeigs, eigvec = eigsh(H, k = 1)
    print(f"GPU {device}", maxeigs)

    # eigs is the max eigenvalue

    # Assuming min eig is negative,
    # shifting the spectrum by 0.6*eigs will make the min eigenvalue to the max one (in absolute sense)
    shifted = 0.6 * maxeigs

    def min_hess_transform(v):
        return hess_transform(v) - shifted * v
    
    H = LinearOperator((param_len, param_len), matvec=min_hess_transform)
    mineigs = eigsh(H, k = 1, return_eigenvectors=False)
    mineigs += shifted
    
    print(f"GPU {device}", mineigs)
    #assert mineigs < maxeigs, "Min eig more than maxeig"
    if(maxeigs < mineigs):
        maxeigs, mineigs = mineigs, maxeigs
    return torch.tensor(maxeigs), torch.tensor(mineigs)

def vis(rank, model, dirn1, dirn2, criterion, steps, indices, output, args):

    model.to(rank)
    vis_model = copy.deepcopy(model)
    dirn1.to(rank)
    dirn2.to(rank)
    args.batch_size = 256
    train_loader, _ = load_dataset(args, False)
    # print(vis_model.parameters().is_cuda())
    for s, step in enumerate(steps):
        # idx is [a, b]
        a, b = step
        for i, d1, d2, k in zip(model.parameters(), dirn1.parameters(), dirn2.parameters(), vis_model.parameters()):
                k.data = i.data + a*d1.data + b*d2.data
        max_eig, min_eig = give_minmax_eigs(train_loader, vis_model, criterion, rank)
        print(f"GPU {rank} : Min Eig {min_eig}, Max Eig {max_eig}")
        output[indices[s], 0] = min_eig
        output[indices[s], 1] = max_eig

def model2vec(d):
    v_ten = []
    for p in d.parameters():
        v_ten.append(p.reshape(-1))
    return torch.cat(v_ten)

def dirn_dot(d1, d2):
    v1 = model2vec(d1)
    v2 = model2vec(d2)
    return torch.sum(v1*v2)/(torch.linalg.norm(v1) * torch.linalg.norm(v2))

if __name__ == "__main__":
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nprocs = torch.cuda.device_count()
    workers = []

    model = load_model_with_weights(args.weight_path, 'cpu')
    criterion = nn.CrossEntropyLoss()

    # dirn1 = give_model(args)
    # # dirn1.to(device)
    # for param, m_param in zip(dirn1.parameters(), model.parameters()):
    #     if(len(m_param.shape) == 1):
    #         param = torch.zeros_like(m_param)
    #         continue
    #     param.data = torch.randn_like(param.data)
    #     param.data = param.data / torch.linalg.norm(param.data)
    #     param.data *= torch.linalg.norm(m_param)

    # dirn2 = give_model(args)
    # # dirn2.to(device)
    # for param, m_param in zip(dirn2.parameters(), model.parameters()):
    #     if(len(m_param.shape) == 1):
    #         param = torch.zeros_like(m_param)
    #         continue
    #     param.data = torch.randn_like(param.data)
    #     param.data = param.data / torch.linalg.norm(param.data)
    #     param.data *= torch.linalg.norm(m_param)

    directions = torch.load(args.dirn_file)
    dirn1, dirn2 = directions['dirn1'], directions['dirn2']
    print(dirn_dot(dirn1, dirn2))

    alpha = torch.linspace(-1, 1, args.range)
    beta = torch.linspace(-1, 1, args.range)
    mesh_x, mesh_y = torch.meshgrid(alpha, beta)
    mesh = torch.cat([mesh_x.unsqueeze(0), mesh_y.unsqueeze(0)], 0).permute(1, 2, 0).reshape(-1, 2)
    
    num_per_proc = mesh.shape[0] // nprocs
    steps = [mesh[i*num_per_proc:(i+1)*num_per_proc, :] for i in range(nprocs-1)]
    steps.append(mesh[(nprocs-1)*(num_per_proc):, :])
    indices = [torch.arange(i*num_per_proc, (i+1)*num_per_proc) for i in range(nprocs-1)]
    indices.append(torch.arange((nprocs-1)*num_per_proc, mesh.shape[0]))

    # Min And Max Eigenvalues
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
    np.save("2dsave.npy", np.array([[output[..., 0]], [output[..., 1]], [mesh_x.numpy()], [mesh_y.numpy()]]))
