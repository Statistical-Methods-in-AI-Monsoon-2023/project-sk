import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Plot loss and accuracy')
parser.add_argument('--file', type=str)
args = parser.parse_args()

# loss = np.load('loss.npy', allow_pickle=True)
# accs = np.load('accs.npy', allow_pickle=True)
loss, accs, x, y = np.load(args.file, allow_pickle=True)
loss = loss[0]
x = x[0]
y = y[0]
accs = accs[0]

model_unique_id = args.file.replace('.npy', '')

# mkdir results/plots/
import os
os.makedirs('results/plots/', exist_ok=True)

ax = plt.axes(projection='3d')
ax.plot_surface(x, y, loss)
# plt.show()
plt.savefig(f'results/plots/loss-{model_unique_id}.png')

ax = plt.axes(projection='3d')
ax.plot_surface(x, y, accs)
# plt.show()
plt.savefig(f'results/plots/acc-{model_unique_id}.png')
