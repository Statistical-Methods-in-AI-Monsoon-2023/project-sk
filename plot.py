import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Plot loss and accuracy')
parser.add_argument('--file', type=str)
parser.add_argument('--show', action='store_true')
parser.add_argument('--json', action='store_true')
args = parser.parse_args()

# loss = np.load('loss.npy', allow_pickle=True)
# accs = np.load('accs.npy', allow_pickle=True)
array = np.load(args.file, allow_pickle=True)
loss, accs, x, y = array
print(array.shape)
print(loss.shape, accs.shape, x.shape, y.shape)
loss = loss[0]
x = x[0]
y = y[0]
accs = accs[0]

# replace nan with 0
loss = np.nan_to_num(loss,posinf=0,neginf=0)
accs = np.nan_to_num(accs,posinf=0,neginf=0)

# log scale
loss = np.log(loss)
accs = np.log(accs)

# print(loss.min(), loss.max())

def to_xyz_array(arr):
    array = []
    for i in range(len(x)):
        row=[]
        for j in range(len(y)):
            row.append({
                'x': float(x[i,0]),
                'y': float(y[0,j]),
                'z': float(arr[i][j])
            })
        array.append(row)
    return array

# take filename from path
model_unique_id = args.file.split('/')[-1].split('.')[0]

# mkdir results/plots/
import os
os.makedirs('results/plots/', exist_ok=True)
os.makedirs('results/plot_json/', exist_ok=True)

if args.json:
    import json
    json.dump({
        'loss': to_xyz_array(loss),
        'acc': to_xyz_array(accs)
    }, open(f'results/plot_json/{model_unique_id}.json', 'w'), indent=2)

ax = plt.axes(projection='3d')
ax.plot_surface(x, y, loss)
if args.show:
    plt.show()
else:
    plt.savefig(f'results/plots/loss-{model_unique_id}.png')

ax = plt.axes(projection='3d')
ax.plot_surface(x, y, accs)
if args.show:
    plt.show()
else:
    plt.savefig(f'results/plots/acc-{model_unique_id}.png')
