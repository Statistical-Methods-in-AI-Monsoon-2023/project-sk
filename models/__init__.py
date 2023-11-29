from datetime import datetime

def give_model(args):
    from models.resnet import ResNet
    from models.vgg import VGG
    from models.cnn import CNN
    from models.vit import VIT
    from models.mlp import MLP
    dataset = args.dataset
    args = args.model.split("-", 1)
    name = args[0]
    margs = args[1] if len(args) > 1 else ""
    if name == "resnet":
        return ResNet(margs, dataset=dataset)
    elif name == "vgg":
        return VGG(margs, dataset=dataset)
    elif name == "cnn":
        return CNN(dataset)
    elif name == 'vit':
        return VIT(dataset)
    elif name == "mlp":
        return MLP()
    else:
        raise ValueError("Invalid model name")

def gen_unique_id(args):
    # generate a string from args
    unique_id = ""
    argslist=vars(args).items()
    argslist=sorted(argslist)
    for k,v in argslist:
        if(k == "weight_path"):
            continue
        unique_id += f"{k}:{v}_"
    unique_id = unique_id[:-1]
    # unique_id+=f"date:{datetime.now().strftime('%d-%m-%Y-%H.%M.%S')}"
    return unique_id

def gen_unique_id_from_filename(filename):
    # take filename from path
    model_unique_id = filename.split('/')[-1].split('.')[0]
    return model_unique_id