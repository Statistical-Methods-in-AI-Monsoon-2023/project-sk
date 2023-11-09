from models.resnet import ResNet
from models.vgg import VGG


def give_model(args):
    args = args.model.split("-", 1)
    name = args[0]
    margs = args[1] if len(args) > 1 else ""
    if name == "resnet":
        return ResNet(margs)
    elif name == "vgg":
        return VGG(margs)
    else:
        raise ValueError("Invalid model name")