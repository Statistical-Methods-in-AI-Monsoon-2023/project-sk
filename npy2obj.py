import numpy as np
import tripy
from collections import Counter
import argparse

# Create NumPy arrays for X, Y, and Z data (example data)

parser = argparse.ArgumentParser(description='Convert npy file to VTK format')
parser.add_argument('--file', type=str)
parser.add_argument('--show', action='store_true')
args = parser.parse_args()

# Create NumPy arrays for X, Y, and Z data (example data)
loss, accs, x, y = np.load(args.file, allow_pickle=True)

loss = loss[0]
X = x[0]
Y = y[0]
accs = accs[0]

# replace nan with very high value
loss = np.nan_to_num(loss, nan = 1e50)
Z = np.log(loss)

# Convert the X, Y, and Z arrays into a list of (x, y, z) vertices
vertices = [(X[i, j], Y[i, j], Z[i, j]) for i in range(X.shape[0]) for j in range(X.shape[1])]

# Generate the triangles from the vertices using Delaunay triangulation
triangles = tripy.earclip(vertices)

# Create a mapping of vertices to their indices
vertex_indices = {vertex: index + 1 for index, vertex in enumerate(vertices)}

# Create an OBJ file content
obj_content = []

for vertex in vertices:
    obj_content.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}")

for triangle in triangles:
    triangle_indices = [vertex_indices[vertex] for vertex in triangle]
    obj_content.append(f"f {triangle_indices[0]} {triangle_indices[1]} {triangle_indices[2]}")

# Save the OBJ content to a file
model_unique_id = args.file.split('/')[-1].split('.')[0]
obj_filename = f"results/{model_unique_id}.obj"
with open(obj_filename, "w") as obj_file:
    obj_file.write("\n".join(obj_content))

print(f"Data saved as {obj_filename}")
