header = """
#!/bin/bash
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4-00:00:00
#SBATCH --partition=ihub
"""[1:]
cmd_header = """
module load u18/cuda/10.2
module load u18/cudnn/7.6.5-cuda-10.2
"""[1:]

import argparse
import os

from models import gen_unique_id

parser = argparse.ArgumentParser(description="Make SBATCH v1.0")
parser.add_argument("--account", "-A", "-a", type=str, default="neuro")
args = parser.parse_args()

hyperparameters = {
    "model": [
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
    ],
    "batch_size": [
        128,
        1024,
    ],
    "lr": [
        0.1,
        0.01,
        0.001,
        0.0001,
    ],
    "epochs": [2000],
    "weight_decay": [
        0,
        5e-4,
    ],
    "optimizer": [
        "sgd",
        "adam",
    ],
    "save_every": [1]
}
os.makedirs("sbatch", exist_ok=True)
os.makedirs("logs", exist_ok=True)

from itertools import product

combo = list(product(*hyperparameters.values()))
print("Total number of jobs:", len(combo))

for hyperparam in combo:
    argsdict = dict(zip(hyperparameters.keys(), hyperparam))
    argstring = " ".join(
        [f"--{k} {v}" for k, v in zip(hyperparameters.keys(), hyperparam)])

    args_namespace = argparse.Namespace(**argsdict)
    unique_id = gen_unique_id(args_namespace)
    weights_path = f"weights/{unique_id}.pt"
    filename = f"sbatch/{unique_id}.sh"
    logfile = f"logs/{unique_id}.out"

    model = argsdict["model"]
    cmds = [
        f"mkdir -p logs",
        f"python3 mp.py {argstring}",
        f"python3 vis_mp.py --weight_path {weights_path} --model {model} --range {50}"
    ]
    with open(filename, "w") as f:
        f.write(header)
        f.write(f"#SBATCH --output={logfile}\n")
        f.write(cmd_header)
        f.write("\n".join(cmds))
