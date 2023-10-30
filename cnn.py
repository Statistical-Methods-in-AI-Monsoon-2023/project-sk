import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F

class CNN(nn.Module):
    
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(1, filters[0]['num_filters'], filters[0]['kernel_size'])
        self.conv2 = nn.Conv2d(filters[0]['num_filters'], filters[1]['num_filters'], filters[1]['kernel_size'])
        self.maxpool = nn.MaxPool2d(3)
        self.fc = nn.Linear(20, 10)
        self.dropout = nn.Dropout2d(p = 0.01)
        
    def forward(self, x):
        N = x.shape[0]
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.dropout(x).reshape(N, -1)
        x = self.fc(x)
        return x
    
class MNIST(Dataset):
    '''
    Dataset Class for MNIST images
    '''
    def __init__(self, data, labels):
        super().__init__()
        '''
        the data should be of the shape (N, 28, 28) where each element represents the pixel value in the range from [0, 1]
        '''
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        '''
        return the same image for now
        '''
        return (self.data[index], self.labels[index])