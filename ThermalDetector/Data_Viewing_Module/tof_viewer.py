import matplotlib.pyplot as plt
from matplotlib import cm as _pltcolormap
from matplotlib import colors as _pltcolors
import numpy as np
import pandas as pd
import sys
import math


filename = sys.argv[1]
df = pd.read_csv(filename)
df_list = df.values.tolist()
max_index = len(df) - 1
index = 0
print(max_index)
dim = int(math.sqrt(len(df_list[0]) - 2))
colormap = _pltcolors.LinearSegmentedColormap.from_list(
    "", ['#0008e3', '#6aed37', '#e3000f'])
norm = _pltcolors.Normalize(
    vmin=min(min(x[0:(dim*dim)]) for x in df_list), vmax=max(max(x[0:(dim*dim)]) for x in df_list), clip=True)


def plot_heatmap(i):
    data = df_list[i][0:(dim*dim)]
    heat_matrix = []
    for n in range(dim):
        heat_matrix.append(data[n*dim:n*dim+dim])
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
              "]\n\n"+str(df_list[index][dim*dim]))
    plt.colorbar(plt.imshow(heatmaps[index], norm=norm))
    plt.draw()


fig, ax1 = plt.subplots()
im = ax1.imshow(heatmaps[index], norm=norm)
# plt.ylim(-0.5,7.5)
# plt.xlim(-0.5,7.5)
fig.canvas.mpl_connect('key_press_event', on_press)
plt.title("FRAME ["+str(index)+" / "+str(max_index) +
          "]\n\n"+str(df_list[index][dim*dim]))
plt.colorbar(im)
plt.show()
