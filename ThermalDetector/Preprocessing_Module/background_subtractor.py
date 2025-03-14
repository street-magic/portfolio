import numpy as np
import sys
import pandas as pd
import math
import sys
import pathlib
import os

AVG = [0.4412535,  0.20667809, -0.61849255, -0.45215652, -1.38443582,
       -0.99791856, -0.46623708, -0.24093715,  0.16735466, -0.06987688,
       -0.13519916, -0.40138724, -0.54655695, -1.03330881, -1.14087943,
       -0.49863535, -0.55127682, -0.03893927,  0.15762419,  0.08672506,
       -0.06094022, -0.145343, -0.20349913, -0.01384304, -1.53851646,
       -0.32619091,  0.15765641, -0.03478394,  0.06214772, -0.04530537,
       -0.14474446, -0.38655408, -0.8346859, -0.41594124, -0.09855538,
       0.09021421,  0.21530864, -0.12777397,  0.13131615,  0.34486676,
       -0.16162728, -0.28541327, -0.00158359,  0.40434041,  0.03817224,
       0.0610977, -0.19110897,  0.25794323,  0.91636137,  0.07913269,
       -0.01713727,  0.27365723,  0.09179505, -0.16565811,  0.34350621,
       0.69421214,  1.05304483,  1.4004424,  1.05112481,  0.95599601,
       0.96484039,  0.92457198,  1.02378982,  1.1802693]


def remove_bg(frame):
    vals = np.array(frame)

    hist, bin_edges = np.histogram(vals, bins=np.linspace(
        math.ceil(min(vals)), math.floor(max(vals)), 8))
    bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
    AVG_bg = bin_centers[np.argmax(hist)]
    diff = (vals - (AVG_bg + AVG)).reshape(8, 8)
    diff[diff < 1.75] = 0
    return diff


######
rootdir = sys.argv[1]
print(rootdir)
BG_removed_data = None
flag = True

prev_cwd = pathlib.Path.cwd()
os.chdir(rootdir)
for path in pathlib.Path().cwd().iterdir():
    if path.is_dir():
        if ('NOP-1' in path.name) or ('NOP-2' in path.name):
            raw_df = pd.read_csv(path / "EYE_old.csv")
            raw_df_nolabels = raw_df.drop(
                ['received_time', 'NOP', 'bed_present', 'bed_occupied'], axis=1)

            for p in range(raw_df_nolabels.shape[0]):
                bg_removed = remove_bg(raw_df_nolabels.iloc[p].values)
                if bg_removed.nonzero()[0].size != 0:
                    if "NOP-1" in path.name:
                        bg_removed = np.append(bg_removed, 1)
                    elif "NOP-2" in path.name:
                        bg_removed = np.append(bg_removed, 2)
                    if flag:
                        BG_removed_data = np.expand_dims(
                            bg_removed, axis=0)
                        flag = False
                    else:
                        print(BG_removed_data.shape,
                              " --- ", bg_removed.shape)
                        BG_removed_data = np.concatenate(
                            (BG_removed_data, np.expand_dims(bg_removed, axis=0)), axis=0)
os.chdir(prev_cwd)

pd.DataFrame(BG_removed_data).to_csv(
    rootdir+r"\bg_removed_data.csv", index=False)
