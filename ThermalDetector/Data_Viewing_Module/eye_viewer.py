# Views thermal sensor recorded data

import matplotlib.pyplot as plt
from matplotlib import cm as _pltcolormap
from matplotlib import colors as _pltcolors
import numpy as np
import pandas as pd
import sys

filename = sys.argv[1]
df = pd.read_csv(filename)
df_list = df.values.tolist()
max_index = len(df) - 1
index = 0
print(max_index)
colormap = _pltcolors.LinearSegmentedColormap.from_list(
    "", ['#0008e3', '#6aed37', '#e3000f'])
norm = _pltcolors.Normalize(
    vmin=19, vmax=26, clip=True)


def plot_heatmap(i):
    data = df_list[i][0:64]
    heat_matrix = []
    for n in range(8):
        heat_matrix.append(data[n*8:n*8+8])
    return heat_matrix


heatmaps = []
for i in range(max_index + 1):
    heatmaps.append(plot_heatmap(i))


def on_press(event):
    global index
    if event.key == 'right':
        if index < max_index:
            index += 1
    if event.key == 'left':
        if index > 0:
            index -= 1
    plt.clf()
    plt.title("FRAME ["+str(index)+" / "+str(max_index) +
              "]\n\n"+str(df_list[index][64]))
    plt.colorbar(plt.imshow(heatmaps[index], cmap=colormap, norm=norm))
    plt.draw()


fig, ax1 = plt.subplots()
im = ax1.imshow(heatmaps[index], cmap=colormap, norm=norm)
# plt.ylim(-0.5,7.5)
# plt.xlim(-0.5,7.5)
fig.canvas.mpl_connect('key_press_event', on_press)
plt.title("FRAME ["+str(index)+" / "+str(max_index) +
          "]\n\n"+str(df_list[index][64]))
plt.colorbar(im)
plt.show()
