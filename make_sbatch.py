"""NOTE: Create a file ./sbatch/sbatch.sh with ###HERE### as a placeholder for the commands to run"""
sbatch_file = open("sbatch/sbatch.sh", "r").read()

import argparse
import os
from collections import OrderedDict

from models import gen_unique_id

parser = argparse.ArgumentParser(description="Make SBATCH v1.0")
parser.add_argument("--account", "-A", "-a", type=str, default="neuro")
parser.add_argument("--single_file", "-s", action="store_true")
args = parser.parse_args()

hyperparameters = OrderedDict()


hyperparameters["model"] = [
        "resnet-noshort",
        "resnet-short",
        "vgg-11-bn",
        # "vgg-13-bn",
        # "vgg-16-bn",
        "vgg-19-bn",
        "vgg-11",
        # "vgg-13",
        # "vgg-16",
        "vgg-19",
    ]
hyperparameters["batch_size"]= [
        128,
        1024,
    ]
hyperparameters[ "lr"]=[
        0.1,
        # 0.01,
        # 0.001,
        # 0.0001,
    ]
hyperparameters["epochs"]= [2000]
hyperparameters["weight_decay"]= [
        0,
        5e-4,
    ]
hyperparameters["optimizer"]= [
        "sgd",
        "adam",
    ]
hyperparameters["save_every"]= [1]
os.makedirs("sbatch", exist_ok=True)
os.makedirs("logs", exist_ok=True)

from itertools import product

combo = list(product(*hyperparameters.values()))
print("Total number of jobs:", len(combo))


combo = [
    # (model, batch_size, lr, epochs, weight_decay, optimizer, save_every),
    # vgg
    ("vgg-9", 128, 0.1, 300, 0, "sgd", 5),
    ("vgg-9", 1024, 0.1, 300, 0, "sgd",5),
    ("vgg-9", 128, 0.1, 300, 5e-4, "sgd", 5),
    ("vgg-9", 1024, 0.1, 300, 5e-4, "sgd",5),
    ("vgg-9", 128, 0.1, 300, 0, "adam", 5),
    ("vgg-9", 1024, 0.1, 300, 0, "adam",5),
    ("vgg-9", 128, 0.1, 300, 5e-4, "adam", 5),
    ("vgg-9", 1024, 0.1, 300, 5e-4, "adam",5),
    
    ("vgg-19", 128, 0.1, 300, 5e-4, "sgd",-1),

    # resnet
    ("resnet-56-noshort-1", 128, 0.1, 300, 5e-4, "sgd", 1),
    ("resnet-56-short-1", 128, 0.1, 300, 5e-4, "sgd", 1),
    ("resnet-56-noshort-2", 128, 0.1, 300, 5e-4, "sgd", -1),
    ("resnet-56-short-2", 128, 0.1, 300, 5e-4, "sgd", -1),
    ("resnet-56-noshort-4", 128, 0.1, 300, 5e-4, "sgd", -1),
    ("resnet-56-short-4", 128, 0.1, 300, 5e-4, "sgd", -1),
    ("resnet-56-noshort-8", 128, 0.1, 300, 5e-4, "sgd", -1),
    ("resnet-56-short-8", 128, 0.1, 300, 5e-4, "sgd", -1),

    ("resnet-20-short-1", 128, 0.1, 300, 5e-4, "sgd", 1),
    ("resnet-20-noshort-1", 128, 0.1, 300, 5e-4, "sgd", 1),
]

if args.single_file:
    filename = "sbatch/sbatch-singlefile.sh"

for hyperparam in combo:
    argsdict = dict(zip(hyperparameters.keys(), hyperparam))
    argstring = " ".join(
        [f"--{k} {v}" for k, v in zip(hyperparameters.keys(), hyperparam)])

    args_namespace = argparse.Namespace(**argsdict)
    unique_id = gen_unique_id(args_namespace)
    weights_path = f"weights/{unique_id}.pt"
    if not args.single_file:
        filename = f"sbatch/{unique_id}.sh"
    logfile = f"logs/{unique_id}.out"

    model = argsdict["model"]
    cmds = [
        f"# jobname: {filename}",
        f"mkdir -p logs", f"python3 mp.py {argstring}",
        f"python3 vis_mp.py --weight_path {weights_path} --model {model} --range {50}"
    ]

    new_sbatch_file = sbatch_file.replace("###HERE###", "\n".join(cmds))

    if args.single_file:
        with open(filename, "a") as f:
            f.write(new_sbatch_file)
    else:
        with open(filename, "w") as f:
            f.write(new_sbatch_file)
