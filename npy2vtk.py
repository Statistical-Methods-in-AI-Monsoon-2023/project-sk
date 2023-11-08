import numpy as np
import pyvista as pv
import argparse

parser = argparse.ArgumentParser(description='Convert npy file to VTK format')
parser.add_argument('--file', type=str)
parser.add_argument('--show', action='store_true')
args = parser.parse_args()

# Create NumPy arrays for X, Y, and Z data (example data)
loss, accs, x, y = np.load(args.file, allow_pickle=True)

loss = loss[0]
x = x[0]
y = y[0]
accs = accs[0]

# replace nan with very high value
loss = np.nan_to_num(loss, nan = 1e50)
loss = np.log(loss)

# Create a PyVista structured grid from the NumPy data
grid = pv.StructuredGrid(x, y, loss)

model_unique_id = args.file.split('/')[-1].split('.')[0]
# Save the structured grid to a VTK file
vtk_filename = f"results/{model_unique_id}.vtk"
grid.save(vtk_filename)

print(f"Data saved as {vtk_filename}")
