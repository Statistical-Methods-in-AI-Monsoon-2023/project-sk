import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models import give_model, gen_unique_id
from data import load_cifar10
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import time
import argparse
import math

parser = argparse.ArgumentParser(description='Train a model on CIFAR10')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
parser.add_argument('--save_every', type=int, default=-1, help='Save every save_every iterations')
parser.add_argument('--model', type=str, help='Name of the model',required=True)


def setup(rank, args):
    # Ininitalizes the process_group and makes this process a part of that group. 
    # MASTER_ADDR and MASTER_PORT are required for the processes to communicate to the master node

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '3003'
    init_process_group('nccl', rank=rank, world_size=args.world_size)

def calc_weight_norm(model):
    w2 = sum([torch.sum(param ** 2).item() for param in model.parameters()])
    return math.sqrt(w2)

def train(rank, args):

    setup(rank, args)

    train_loader, test_loader= load_cifar10(int(args.batch_size), 2, distributed=True)

    # Create the ResNet model and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = give_model(args).to(device)
    model_unique_id = gen_unique_id(args)
    # Create a DDP instance
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Training loop
    num_epochs = args.epochs
    count = 0
    t1 = time.time()
    if(rank == 0):
        losses = []
        weight_norm = []

    if(args.save_every != -1):
        os.makedirs(f'weights/{model_unique_id}', exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        count = 0 
        for batch_idx, (data, target) in enumerate(train_loader):
            count += 1
            optimizer.zero_grad()
            data = data.to(rank)
            target = target.to(rank)
            output = model(data)
            loss = F.cross_entropy(output, target)
            if(rank == 0):
                losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if(rank == 0):
                losses.append(loss.item())
            
            if batch_idx % 100 == 0:
                print(f'GPU {rank} : Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
        
        if(rank == 0):
            weight_norm.append(calc_weight_norm(model))
            if(epoch % args.save_every == 0 and args.save_every != -1):
                torch.save(model.module, f"weights/{model_unique_id}/{epoch}.pt")

    if(rank == 0):
        torch.save({"model" : model.module, "losses" : losses, "weight_norm" : weight_norm}, f"weights/{model_unique_id}.pt")
    
    # Test the model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(rank)
            target = target.to(rank)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum()
    torch.distributed.all_reduce(tensor=correct, op=torch.distributed.ReduceOp.SUM)
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy}%')
    destroy_process_group()

if __name__ == '__main__':
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    args.world_size = world_size
    print(world_size)
    print(args)
    losses = torch.zeros((args.epochs))
    weight_norm = torch.zeros((args.epochs))
    losses.share_memory_()
    weight_norm.share_memory_()
    # Creates (world_size) number of processes
    mp.spawn(train, args=[args], nprocs=world_size)
    # train()
