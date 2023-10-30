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

def give_model(path):
    model_init = CNN()
    model_init = torch.load(path, map_location=torch.device('cpu'))
    model_init.eval()

    return model_init


# Define data transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Load and preprocess the MNIST dataset
test_dataset = MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=32)

model1 = give_model('weights/m1.pt')
model2 = give_model('weights/m2.pt')

# Uncomment for printing the parameters of the model

# for name, param in model2.named_parameters():
    # print(name)
    # print(param.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

@torch.no_grad()
def give_loss_acc(dataloader, model, criterion):
    loss = 0
    num = 0
    corr = 0
    for inputs, labels in dataloader:
        num += inputs.shape[0]
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss += criterion(outputs, labels)
        corr += torch.where(labels == torch.argmax(outputs, -1))[0].shape[0]

    acc = corr / 10000
    return loss, acc


def interpolate(dataloader, criterion, model1, model2):
    stepsize = 20
    alpha = torch.linspace(0, 1, stepsize)
    model = CNN()
    losses = []
    accs = []
    for a in alpha:
        for i, j, k in zip(model1.parameters(), model2.parameters(), model.parameters()):
            k.data = (i.data * a + j.data * (1 - a))

        loss, acc = give_loss_acc(dataloader, model, criterion)
        print(loss)
        print(acc)
        losses.append(loss)
        accs.append(acc)
    
    return losses, accs

def plot_loss_acc(model1, model2, criterion, dataloader):
    loss, accs = interpolate(dataloader, criterion, model1, model2)
    plt.plot(loss)
    plt.show()
    plt.plot(accs)
    plt.show()


plot_loss_acc(model1, model2, criterion, test_loader)