import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3)
        self.conv2 = nn.Conv2d(5, 5, 3)
        self.maxpool = nn.MaxPool2d(2)
        self.fc = nn.Linear(125, 10)
        self.dropout = nn.Dropout2d(p = 0.01)
        
    def forward(self, x):
        N = x.shape[0]
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.dropout(x).reshape(N, -1)
        x = self.fc(x)
        return F.sigmoid(x)
    
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