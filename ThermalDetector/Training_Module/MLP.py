import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import keras.backend as K


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
    plt.title('Confusion Matrix for MLP Model\n')
    plt.savefig("CM-MLP.png")


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


class MLP():

    def __init__(self):
        self.X = None
        self.Y = None
        self.path = r"C:\Users\asus vivobook 14\Downloads\G06_Hospital\40_Realisation\10_Prototype_1\Data_collector_and_Visualizer\classifier"
        self.epoch = 10
        self.model = tf.keras.Sequential()

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
        self.Y_train = tf.keras.utils.to_categorical(Y_train-1)
        self.Y_test = tf.keras.utils.to_categorical(Y_test-1)

   # Building an MLP

    def build(self):

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(
            20, activation='relu', input_dim=7))
        self.model.add(tf.keras.layers.Dense(40, activation='relu'))
        self.model.add(tf.keras.layers.Dense(20, activation='relu'))
        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))

        # Compile the model
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy', f1_m, precision_m, recall_m])

# model training

    def train(self, epoch, batch):
        self.epoch = epoch
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=10)
        self.model.fit(self.X_train, self.Y_train,
                       epochs=self.epoch, batch_size=batch, callbacks=[early_stopping])

        loss, accuracy, f1_score, precision, recall = self.model.evaluate(
            self.X_test, self.Y_test, verbose=0)
        print(
            f"loss - {loss} | accuracy - {accuracy} | f1 score - {f1_score} | precision - {precision}")

        y_pred_test = self.model.predict(self.X_test)
        pred = []
        for p in y_pred_test:
            if p[0] >= p[1]:
                pred += [1]
            else:
                pred += [2]

        new_y = []
        for p in self.Y_test:
            if p[0] >= p[1]:
                new_y += [1]
            else:
                new_y += [2]

        plot_CM(new_y, pred)


# exporting model as a .h5 file

    def save(self):
        self.model.save(f"{self.path}\MLP_model_e_"+str(self.epoch)+".h5")

    def save(self, path):
        self.path = path
        self.model.save(f"{self.path}\model_MLP1.h5")

    def load(self):
        return tf.keras.models.load_model(f"{self.path}\MLP_model_e_400.h5")

    def load(self, path, name):
        return tf.keras.models.load_model(f"{self.path}\\{name}")
