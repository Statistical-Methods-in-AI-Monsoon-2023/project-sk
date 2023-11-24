from models.resnet import ResNet
from models.vgg import VGG
from models.cnn import CNN
from models.vit import VIT

from datetime import datetime

def give_model(args):
    dataset = args.dataset
    args = args.model.split("-", 1)
    name = args[0]
    margs = args[1] if len(args) > 1 else ""
    if name == "resnet":
        return ResNet(margs, dataset=dataset)
    elif name == "vgg":
        return VGG(margs)
    elif name == "cnn":
        return CNN(dataset)
    elif name == 'vit':
        return VIT(dataset)
    else:
        raise ValueError("Invalid model name")

def gen_unique_id(args):
    # generate a string from args
    unique_id = ""
    argslist=vars(args).items()
    argslist=sorted(argslist)
    for k,v in argslist:
        unique_id += f"{k}:{v}_"
    unique_id = unique_id[:-1]
    # unique_id+=f"date:{datetime.now().strftime('%d-%m-%Y-%H.%M.%S')}"
    return unique_id