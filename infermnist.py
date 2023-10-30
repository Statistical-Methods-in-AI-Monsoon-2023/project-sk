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
from models.cnn import CNN
from visfuncs  import interpolate

def give_model(path):
    model_init = CNN()
    model_init = torch.load(path, map_location=torch.device('cpu'))
    model_init.eval()

    return model_init

def plot_loss_acc(model1, model2, criterion, dataloader, device):
    loss, accs = interpolate(dataloader, criterion, model1, model2, device, log=True)
    plt.plot(loss)
    plt.show()
    plt.plot(accs)
    plt.show()

# Define data transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Load and preprocess the MNIST dataset
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32)

model1 = give_model('weights/m1.pt')
model2 = give_model('weights/m2.pt')

# Uncomment for printing the parameters of the model

# for name, param in model2.named_parameters():
    # print(name)
    # print(param.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

plot_loss_acc(model1, model2, criterion, test_loader, device)