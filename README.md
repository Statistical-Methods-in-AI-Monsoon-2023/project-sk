[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Zksn1waN)


The repository contains code for the implementation of the paper 'Visualizing Loss Landscapes'.
The directory structure is as follows
data -> would contain the data files
models -> contains the implementation of various models
    -> 2 Layered CNN
    -> MLP
    -> ResNet{18,56,110}-{Short,NoShort}
    -> VGG{11, 13, 16, 19}
    -> VIT
results -> would contain the npy files along with the plots obtained
sbatch -> contains the sbatch scripts for running on ada
viz -> contains the loss landscape visualization code using 3js
weights -> contatins the weights of the models trained 

data.py -> Contains the dataloader for various datasets -> CIFAR10, MNIST, XOR, IDD
make_sbatch.py -> Helper Script for making the sbatch scripts

mp.py -> The training script (parallelized on any number of gpus)
The arguments are
    -> epochs -> The nubmer of epochs to train for
    -> lr -> The initial learning rate of training
    -> weight_decay -> Weight decay for the optimizer 
    -> optimizer -> Optimizer (SGD|ADAM)
    -> save_every -> Save after every save_every epochs (-1 for saving just the final weights)
    -> model -> The model to train (CNN|VIT|ResNet|ResNet-Noshort|MLP) (resnet model should be of the form
    resnet-<depth>-<short/noshort>-<filters_mag> where depth is the nubmer of layers, short/noshort indicates whether to use skip connections or not, filter_mag is the multiplication factor for the number of filters)
    -> dataset -> The dataset to train the model on (CIFAR10|MNIST)

An example command is
```
    python3 mp.py --epochs 300 --lr 0.1 --weight_decay 5e-4 optimizer sgd save_every -1 --model resnet-56-noshort-1 --dataset cifar10
```

vis_mp.py -> The visualizing scipt for the models, (perturbes models parameters in 2 perpendicular
directions, and records the loss and accuracy at each point)
The arguments are
    weight_path -> The path to the trained weights
    range -> The range of the grid. The size of the grid is (range, range)
    dataset -> The dataset on which the model was trained on

An example command is
```
    python3 vis_mp.py --weight_path ./weights/resnet-300-lr-0.01.pt --range 20 --dataset cifar10
```

vis_hessian.py -> The script takes in the model weights and plots the hessian matrix at the points according 
to the direction returned by vis_mp.py

The arguments are
    weight_path -> The path to the trained weights
    range -> The range of the grid. The size of the grid is (range, range)
    dataset -> The dataset on which the model was trained on
    dirn_file -> The path to the directions file
    model -> The model corresponding to the weights
    
An example command is
```
    python3 vis_hessian.py --weight_path ./weights/resnet-300-lr-0.01.pt --range 20 --dataset cifar10 --dirn_file ./dirns-resnet56-noshort-300-lr-0.01.pt --model resnet-56-noshort-1
```

vis_lininterp.py -> The script interpolates between 2 models of the same architecture
The arguments are 
    weight_path1 -> The path to the first weights file
    weight_path2 -> The path to the second weight file
    model -> The model corresponding to the weight file
    range -> The resolution of the interpolation (from -1 to 1.5, the number of discrete values between them is equal to the range).

An example command is
```
    python3 vis_lininterp.py --weight_path1 ./weights/resnet-300-lr-0.01.pt --weight_path2 ./weights/resnet-300-lr-0.001.pt --range 20 --model resnet-56-noshort-1
```