from models.resnet import ResNet
from models.vgg import VGG


def give_model(args):
    name = args.mname
    margs = args.margs
    if name == "resnet":
        return ResNet(margs)
    elif name == "vgg":
        return VGG(margs)
    else:
        raise ValueError("Invalid model name")