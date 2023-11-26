import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        assert dataset == "xor", "MLP for dataset xor only"
        # if(dataset == 'cifar10'):
            # self.input_size = (3, 32, 32)
        # elif(dataset == 'mnist'):
            # self.input_size = (1, 28, 28)
        self.lin1 = nn.Linear(2, 1)
        
    def forward(self, x):
        x = self.lin1(x)
        return F.sigmoid(x)