import torch
import torch.nn as nn
import torch.nn.functional as F

class LambdaLayer(nn.Module):
    """
      Identity mapping between ResNet blocks with diffrenet size feature map
    """
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.planes = planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if option == 'A':
                """
                For CIFAR10 experiment, ResNet paper uses option A.
                """
            #    self.shortcut = LambdaLayer(lambda x:
            #                           F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
                self.shortcut = None
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out = F.relu(self.conv1(x))
        # out = (self.conv2(out))
        if(self.shortcut == None):
            out += F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes//4, self.planes//4), "constant", 0)
        else:
            out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class BasicBlockNoShort(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockNoShort, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out = F.relu(self.conv1(x))
        # out = (self.conv2(out))

        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, margs="noshort", dataset="cifar10", num_blocks=[9,9,9], num_classes=10):
        super(ResNet, self).__init__()
        # margs of the form size-short/noshort-filters_mag
        self.in_planes = 16
        
        if(dataset == "cifar10"):
            self.channel_num = 3
            self.final_avg_pool = 8
        elif(dataset == "mnist"):
            self.channel_num = 1
            self.final_avg_pool = 7

        args = margs.split('-')

        resnet_size = int(args[0])
        self.block_type = args[1]
        filters_mag = int(args[2])

        if(resnet_size == 18):
            num_blocks = [3, 3, 3]
        elif(resnet_size == 56):
            num_blocks = [9, 9, 9]
        elif(resnet_size == 110):
            num_blocks = [18, 18, 18]

        #self.avg_pool = nn.AvgPool2d(self.final_avg_pool)
        if self.block_type == "short":
            print("short")
            block = BasicBlock
        elif self.block_type == "noshort":
            block = BasicBlockNoShort
        else:
            raise ValueError("block_type must be one of 'short' or 'noshort'")

        self.conv1  = nn.Conv2d(self.channel_num, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16*filters_mag, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32*filters_mag, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64*filters_mag, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*filters_mag*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.avg_pool(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def get_unique_id(self):
        return f"resnet-{self.block_type}"
