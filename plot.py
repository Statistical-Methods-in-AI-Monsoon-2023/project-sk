import matplotlib.pyplot as plt
import numpy as np

# loss = np.load('loss.npy', allow_pickle=True)
# accs = np.load('accs.npy', allow_pickle=True)
loss, accs, x, y = np.load('2d.npy', allow_pickle=True)
loss = loss[0]
x = x[0]
y = y[0]
accs = accs[0]

ax = plt.axes(projection='3d')
ax.plot_surface(x, y, loss)
plt.show()
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, accs)
plt.show()