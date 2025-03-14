import sys
import numpy as np
import pandas as pd
import CNN
import MLP
import RF

csv_path = sys.argv[1]
model_name = sys.argv[2]
out_path = sys.argv[3]

if model_name == "CNN":
    data = pd.read_csv(csv_path)
    cnn_model = CNN.CNN()
    X = data.drop("64", axis=1)
    Y = data["64"]
    cnn_model.reshape_data(X.to_numpy(), Y.to_numpy())
    cnn_model.build_CNN()
    cnn_model.train_CNN(100, 32)
    cnn_model.save_CNN(out_path)

else:
    data = pd.read_csv(csv_path)
    X = data.drop("7", axis=1)
    Y = data["7"]
    model = None
    if model_name == "MLP":
        model = MLP.MLP()
    elif model_name == "RF":
        model = RF.RF()
    model.reshape_data(X.to_numpy(), Y.to_numpy())
    model.build()
    if model_name == "RF":
        model.train()
    else:
        model.train(100, 32)
    model.save(out_path)
