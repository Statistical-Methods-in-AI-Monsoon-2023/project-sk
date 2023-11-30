import matplotlib.pyplot as plt
import numpy as np

from models import gen_unique_id_from_filename
import argparse
import os

os.makedirs('results/plots/', exist_ok=True)
parser = argparse.ArgumentParser(description='Plot loss and accuracy')
parser.add_argument('--file', type=str)
parser.add_argument('--show', action='store_true')
parser.add_argument('--json', action='store_true')
parser.add_argument('--plot',
                    type=str,
                    default='surface',
                    choices=['surface', 'contour', 'sharp_flat'])
parser.add_argument('--transparent', action='store_true')
args = parser.parse_args()

model_unique_id = gen_unique_id_from_filename(args.file)

array = np.load(args.file, allow_pickle=True)
loss, accs, x, y = array
print(array.shape)
print(loss.shape, accs.shape, x.shape, y.shape)
loss = loss[0]
x = x[0]
y = y[0]
accs = accs[0]

# replace nan with 0
loss = np.nan_to_num(loss, posinf=0, neginf=0)
accs = np.nan_to_num(accs, posinf=0, neginf=0)


def contour_plot():
    loss = loss / 10000
    contour_plot = plt.contour(x, y, loss, levels=20, cmap='viridis')
    plt.clabel(contour_plot, inline=True, fontsize=8)

    # # Add labels and a colorbar
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # plt.title('Contour Plot Example')
    plt.colorbar(contour_plot, label='Z-axis')

    if args.show:
        plt.show()
    else:
        plt.savefig(f'results/plots/contour-plot-{model_unique_id}.png',
                    transparent=args.transparent)


def sharp_flat_plot():
    # twin plots: loss and acc
    fig, ax1 = plt.subplots()
    x = np.arange(len(loss)) / len(loss) * 2.5 - 1.25
    ax1.plot(x, loss[:, 0], 'b-')
    ax1.set_xlabel('interpolation (alpha)')
    ax1.set_ylabel('loss', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(x, accs[:, 0], 'r-')
    ax2.set_ylabel('acc', color='r')
    ax2.tick_params('y', colors='r')

    # get base file name
    import os
    filename = os.path.basename(args.file).replace('.npy', '')

    # keep "model:(...)_" only
    filename = filename.split('model:')[1]
    filename = filename.split('_')[0]

    plt.suptitle(filename)

    # fig.tight_layout()
    if args.show:
        plt.show()
    else:
        plt.savefig(f'results/plots/sharp-flat-plot-{filename}.png',
                    transparent=args.transparent)


def surface_plot():
    # # log scale
    # loss = np.log(loss)
    # accs = np.log(accs)

    # # print(loss.min(), loss.max())

    def to_xyz_array(arr):
        array = []
        for i in range(len(x)):
            row = []
            for j in range(len(y)):
                row.append({
                    'x': float(x[i, 0]),
                    'y': float(y[0, j]),
                    'z': float(arr[i][j])
                })
            array.append(row)
        return array

    if args.json:
        os.makedirs('results/plot_json/', exist_ok=True)
        import json
        json.dump({
            'loss': to_xyz_array(loss),
            'acc': to_xyz_array(accs)
        },
                  open(f'results/plot_json/{model_unique_id}.json', 'w'),
                  indent=2)

    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, loss)
    if args.show:
        plt.show()
    else:
        plt.savefig(f'results/plots/loss-{model_unique_id}.png',
                    transparent=args.transparent)

    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, accs)
    if args.show:
        plt.show()
    else:
        plt.savefig(f'results/plots/acc-{model_unique_id}.png',
                    transparent=args.transparent)


plots = {
    'contour': contour_plot,
    'sharp_flat': sharp_flat_plot,
    'surface': surface_plot
}

plots[args.plot]()
