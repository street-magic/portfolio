import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
    plt.savefig("CM-CNN.png")


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


class CNN():

    def __init__(self):
        self.X = None
        self.Y = None
        self.path = r"C:\Users\asus vivobook 14\Downloads\G06_Hospital\40_Realisation\10_Prototype_1\Data_collector_and_Visualizer\classifier"
        self.epoch = 10
        self.CNN_model = tf.keras.Sequential()

        self.CNN_X_train = None
        self.CNN_X_test = None

        self.CNN_Y_train = None
        self.CNN_Y_test = None

    def reshape_data(self, X, Y):
        self.X = X
        self.Y = Y
        X_shuffle, Y_shuffle = shuffle(X, Y)

        X_train = X_shuffle[0:5669]
        X_test = X_shuffle[5669:]

        Y_train = Y_shuffle[0:5669]
        Y_test = Y_shuffle[5669:]

    # Reshaping the data
        print(X_train.shape)
        self.CNN_X_train = X_train.reshape((X_train.shape[0], 8, 8, 1))
        self.CNN_X_test = X_test.reshape((X_test.shape[0], 8, 8, 1))
        self.CNN_Y_train = tf.keras.utils.to_categorical(Y_train-1)
        self.CNN_Y_test = tf.keras.utils.to_categorical(Y_test-1)

    # Building a CNN
    def build_CNN(self):

        self.CNN_model.add(tf.keras.layers.Conv2D(
            128, (3, 3), activation='relu', input_shape=(8, 8, 1)))
        self.CNN_model.add(tf.keras.layers.Conv2D(
            128, (3, 3), activation='relu'))
        self.CNN_model.add(tf.keras.layers.Dropout(0.5))
        self.CNN_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.CNN_model.add(tf.keras.layers.Conv2D(
            64, (2, 2), activation='relu'))
        self.CNN_model.add(tf.keras.layers.Flatten())
        self.CNN_model.add(tf.keras.layers.Dense(10, activation='relu'))
        self.CNN_model.add(tf.keras.layers.Dense(2, activation='softmax'))
        # Compile the model
        self.CNN_model.compile(optimizer='adam',
                               loss='binary_crossentropy',
                               metrics=['accuracy', f1_m, precision_m, recall_m])

# model training

    def train_CNN(self, epoch, batch):
        self.epoch = epoch
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=10)
        self.CNN_model.fit(self.CNN_X_train, self.CNN_Y_train,
                           epochs=self.epoch, batch_size=batch, callbacks=[early_stopping])

        y_pred_test = self.CNN_model.predict(self.CNN_X_test)

        pred = []
        for p in y_pred_test:
            if p[0] >= p[1]:
                pred += [1]
            else:
                pred += [2]

        new_y = []
        for p in self.CNN_Y_test:
            if p[0] >= p[1]:
                new_y += [1]
            else:
                new_y += [2]

        #plot_CM(new_y, pred)

# exporting model as a .h5 file

    def save_CNN(self):
        self.CNN_model.save(f"{self.path}\CNN_model_e_"+str(self.epoch)+".h5")

    def save_CNN(self, path):
        self.path = path
        self.CNN_model.save(f"{self.path}\model_CNN1.h5")

    def load_model(self):
        return tf.keras.models.load_model(f"{self.path}\CNN_model_e_400.h5")

    def load_model(self, path, name):
        return tf.keras.models.load_model(f"{self.path}\\{name}")
