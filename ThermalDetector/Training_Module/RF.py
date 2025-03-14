#import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


def plot_CM(y_test, y_pred_test):
    # Get and reshape confusion matrix data
    matrix = confusion_matrix(y_test, y_pred_test)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    # Build the plot
    plt.figure(figsize=(65, 40))
    sns.set(font_scale=10)
    sns.heatmap(matrix, annot=True, annot_kws={'size': 230},
                cmap=plt.cm.Greens, linewidths=0.2)

    # Add labels to the plot
    class_names = ['1 person', '2 people']
    tick_marks = np.arange(len(class_names))+0.5
    tick_marks2 = np.arange(len(class_names)) + 0.5
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model\n')
    plt.savefig("CM-RF.png")


class RF():

    def __init__(self):
        self.X = None
        self.Y = None
        self.path = r"C:\Users\asus vivobook 14\Downloads\G06_Hospital\40_Realisation\10_Prototype_1\Data_collector_and_Visualizer\classifier"
        self.model = None

        self.X_train = None
        self.X_test = None

        self.Y_train = None
        self.Y_test = None

    def reshape_data(self, X, Y):
        self.X = X
        self.Y = Y
        X_shuffle, Y_shuffle = shuffle(X, Y)

        self.X_train = X_shuffle[0:5669]
        self.X_test = X_shuffle[5669:]

        Y_train = Y_shuffle[0:5669]
        Y_test = Y_shuffle[5669:]

    # Reshaping the data
        self.Y_train = Y_train.reshape(5669,)
        self.Y_test = Y_test.reshape(1418,)

    # Building a RF

    def build(self, max_depth, random_state):

        self.model = RandomForestClassifier(max_depth, random_state)

    def build(self):

        self.model = RandomForestClassifier(max_depth=5000, random_state=32)


# model training


    def train(self):
        self.model.fit(self.X_train, self.Y_train.ravel())
        y_pred_test = self.model.predict(self.X_test)

        print(accuracy_score(self.Y_test, y_pred_test))
        print(classification_report(self.Y_test, y_pred_test))

        plot_CM(self.Y_test, y_pred_test)
        print("Saved")
        print("Model Trained")


# exporting model as a .h5 file

    def save(self):
        joblib.dump(self.model, f"{path}\model_RF")

    def save(self, path):
        self.path = path
        joblib.dump(self.model, f"{path}\model_RF2")

    def load(self):
        return tf.keras.models.load_model(f"{self.path}\MLP_model_e_400.h5")

    def load(self, path, name):
        return tf.keras.models.load_model(f"{self.path}\\{name}")
