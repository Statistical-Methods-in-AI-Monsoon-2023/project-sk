import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Plot loss and accuracy')
parser.add_argument('--file', type=str)
parser.add_argument('--show', action='store_true')
args = parser.parse_args()

# loss = np.load('loss.npy', allow_pickle=True)
# accs = np.load('accs.npy', allow_pickle=True)
loss, accs, x, y = np.load(args.file, allow_pickle=True)
loss = loss[0]
x = x[0]
y = y[0]
accs = accs[0]

# replace nan with 0
loss = np.nan_to_num(loss)
accs = np.nan_to_num(accs)

# take filename from path
model_unique_id = args.file.split('/')[-1].split('.')[0]

# mkdir results/plots/
import os
os.makedirs('results/plots/', exist_ok=True)

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
