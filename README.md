# Visualizing Loss Landscapes

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Zksn1waN)

## Repository Structure

- **data:** Contains data files.
- **models:** Implementation of various models.
  - 2 Layered CNN
  - MLP
  - ResNet{18, 56, 110}-{Short, NoShort}
  - VGG{11, 13, 16, 19}
  - VIT
- **results:** Contains npy files and plots obtained.
- **sbatch:** Sbatch scripts for running on Ada.
- **viz:** Loss landscape visualization code using 3js.
- **weights:** Contains weights of the trained models.

## Files and Scripts

- **data.py:** Dataloader for various datasets (CIFAR10, MNIST, XOR, IDD).
- **make_sbatch.py:** Helper script for creating sbatch scripts.
- **mp.py:** Training script (parallelized on any number of GPUs).
  - Arguments:
    - epochs: Number of epochs to train for.
    - lr: Initial learning rate of training.
    - weight_decay: Weight decay for the optimizer.
    - optimizer: Optimizer (SGD|ADAM).
    - save_every: Save after every save_every epochs (-1 for saving just the final weights).
    - model: The model to train (CNN|VIT|ResNet|ResNet-Noshort|MLP).
    - dataset: Dataset to train the model on (CIFAR10|MNIST).
  - Example command:
    ```bash
    python3 mp.py --epochs 300 --lr 0.1 --weight_decay 5e-4 optimizer sgd save_every -1 --model resnet-56-noshort-1 --dataset cifar10
    ```
- **vis_mp.py:** Visualization script for the models.
  - Arguments:
    - weight_path: Path to the trained weights.
    - range: Range of the grid. The size of the grid is (range, range).
    - dataset: Dataset on which the model was trained.
  - Example command:
    ```bash
    python3 vis_mp.py --weight_path ./weights/resnet-300-lr-0.01.pt --range 20 --dataset cifar10
    ```
- **vis_hessian.py:** Script for plotting the Hessian matrix.
  - Arguments:
    - weight_path: Path to the trained weights.
    - range: Range of the grid. The size of the grid is (range, range).
    - dataset: Dataset on which the model was trained.
    - dirn_file: Path to the directions file.
    - model: Model corresponding to the weights.
  - Example command:
    ```bash
    python3 vis_hessian.py --weight_path ./weights/resnet-300-lr-0.01.pt --range 20 --dataset cifar10 --dirn_file ./dirns-resnet56-noshort-300-lr-0.01.pt --model resnet-56-noshort-1
    ```
- **vis_lininterp.py:** Script for interpolating between two models of the same architecture.
  - Arguments:
    - weight_path1: Path to the first weights file.
    - weight_path2: Path to the second weight file.
    - model: Model corresponding to the weight file.
    - range: Resolution of the interpolation (from -1 to 1.5).
  - Example command:
    ```bash
    python3 vis_lininterp.py --weight_path1 ./weights/resnet-300-lr-0.01.pt --weight_path2 ./weights/resnet-300-lr-0.001.pt --range 20 --model resnet-56-noshort-1
    ```