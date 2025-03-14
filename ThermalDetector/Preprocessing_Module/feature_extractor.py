import sys
import numpy as np
import pandas as pd
from Preprocessing_Module import feature_extractor

file_path = sys.argv[1]
data_save_path = sys.argv[2]

bg_removed_df = pd.read_csv(file_path)
bg_removed = bg_removed_df.to_numpy()


def lim_check(x, y):
    if (x in range(8) and y in range(8)):
        return True
    return False


def ffill(i, j, A, visited, CC, label, T):
    if (visited[i, j] == 1):
        return
    visited[i, j] = 1
    CC[i, j] = label

    dirs = [[-1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, -1],
            [0, -1],
            [-1, -1]]
    for v in dirs:
        if lim_check(i+v[0], j+v[1]):
            if (abs(A[i, j] - A[i+v[0], j+v[1]]) <= T):
                ffill(i+v[0], j+v[1], A, visited, CC, label, T)


def find_components(A, visited, CC):
    label = 0
    for i in range(8):
        for j in range(8):
            if visited[i, j] == 0:
                label += 1
                ffill(i, j, A, visited, CC, label, 1.7)


def large_comp(frame):
    visited = np.zeros((8, 8))
    CC = np.zeros((8, 8))

    find_components(frame, visited, CC)

    unique, counts = np.unique(CC, return_counts=True)

    comp_list = sorted(list(zip(unique, counts)),
                       key=lambda x: x[1], reverse=True)

    if len(comp_list) <= 1:
        return [0, 0]
    elif len(comp_list) == 2:
        return [comp_list[1][1], 0]

    return [comp_list[1][1], comp_list[2][1]]


flag = True
features = None

for frame in bg_removed:

    num_of_comp = large_comp(frame[:64].reshape(8, 8))
    A = np.array([[np.std(frame[:64]), np.mean(frame[:64]), np.min(frame[np.nonzero(frame[:64])]), max(
        frame[:64]), np.count_nonzero(frame[:64])] + num_of_comp + [frame[64]]])
    if flag:
        features = A
        flag = False
    else:
        features = np.concatenate((features, A), axis=0)

# save extracted features
pd.DataFrame(features).to_csv(
    data_save_path+r"\feature_data.csv", index=False)
