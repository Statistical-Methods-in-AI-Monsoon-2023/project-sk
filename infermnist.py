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

# Define data transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Load and preprocess the MNIST dataset
test_dataset = MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=32)


def give_model(path):
    model_init = CNN()
    model_init = torch.load(path, map_location=torch.device('cpu'))
    model_init.eval()

    return model_init


model1 = give_model('m1.pt')
model2 = give_model('m2.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

corr = 0

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
    model = torch.load("m1.pt", map_location=torch.device('cpu'))
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

def plot_loss(model1, model2, criterion, dataloader):
    loss, accs = interpolate(dataloader, criterion, model1, model2)
    plt.plot(loss)
    plt.show()
    plt.plot(accs)
    plt.show()

for name, param in model2.named_parameters():
    print(name)
    print(param.shape)
plot_loss(model1, model2, criterion, test_loader)

# for i, j in zip(model.parameters(), new_model.parameters()):
    # print(i.data)
    # print(j.data)