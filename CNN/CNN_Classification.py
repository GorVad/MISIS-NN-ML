import numpy as np
import seaborn as sns;
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import pandas as pd
import numpy

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.datasets import cifar100, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

def metric(history, X_train, y_train, X_test, y_test):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()

    predictions_test = np.argmax(model.predict(X_test), axis=1)
    print('Accuracy для тестового множества:', metrics.accuracy_score(y_test, predictions_test))
    print('Precision для тестового множества:', metrics.precision_score(y_test, predictions_test, average='macro'))
    print('Recall для тестового множества:', metrics.recall_score(y_test, predictions_test, average='macro'))
    print('F1-measure для тестового множества:', metrics.f1_score(y_test, predictions_test, average='macro'))

    predictions_train = np.argmax(model.predict(X_train), axis=1)
    print('Accuracy для тренировочного множества:', metrics.accuracy_score(y_train, predictions_train))
    print('Precision для тренировочного множества:', metrics.precision_score(y_train, predictions_train, average='macro'))
    print('Recall для тренировочного множества:', metrics.recall_score(y_train, predictions_train, average='macro'))
    print('F1-measure для тренировочного множества:', metrics.f1_score(y_train, predictions_train, average='macro'))

    cm = confusion_matrix(y_test, predictions_test)
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, cbar=False, fmt='d', annot=True, cmap=plt.cm.Blues)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    print("Матрица ошибок:\n", cm)

# Конфигурация модели
batch_size = 50
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = sparse_categorical_crossentropy
no_classes = 10
no_epochs = 100
optimizer = Adam()
validation_split = 0.2
verbosity = 1
input_shape = (img_width, img_height, img_num_channels)

# Создание модели
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

# Компиляция модели
model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
print(model.summary())


# Обучение не основании CIFAR-10
# (X_train_10, y_train_10), (X_test_10, y_test_10) = cifar10.load_data()
#
# X_train_10 = X_train_10.astype('float32')
# X_test_10 = X_test_10.astype('float32')
#
# X_train_10 = X_train_10 / 255
# X_test_10 = X_test_10 / 255
#
# history = model.fit(X_train_10, y_train_10,
#                     batch_size=batch_size,
#                     epochs=no_epochs,
#                     verbose=verbosity,
#                     validation_split=validation_split,
#                     callbacks=[es])
#
# metric(history, X_train_10, y_train_10, X_test_10, y_test_10)


# Дообучение на суперклассе Household Furniture CIFAR-100
(X_train_100, y_train_100), (X_test_100, y_test_100) = cifar100.load_data(label_mode="coarse")

X_train_100 = X_train_100.astype('float32')
X_test_100 = X_test_100.astype('float32')

X_train_100 = X_train_100 / 255
X_test_100 = X_test_100 / 255

history = model.fit(X_train_100, y_train_100,
                    batch_size=batch_size,
                    epochs=no_epochs,
                    verbose=verbosity,
                    validation_split=validation_split,
                    callbacks=[es])

metric(history, X_train_100, y_train_100, X_test_100, y_test_100)