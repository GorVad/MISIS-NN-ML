import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import tensorflow
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Предварительная обработка изображений
fig, axes = plt.subplots(ncols=7, nrows=3, figsize=(17, 8))
index = 0
for i in range(3):
    for j in range(7):
        axes[i, j].set_title(labels[y_train[index][0]])
        axes[i, j].imshow(X_train[index])
        axes[i, j].get_xaxis().set_visible(False)
        axes[i, j].get_yaxis().set_visible(False)
        index += 1
plt.show()

X_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
X_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

fig, axes = plt.subplots(ncols=7, nrows=3, figsize=(17, 8))
index = 0
for i in range(3):
    for j in range(7):
        axes[i, j].set_title(labels[y_train[index][0]])
        axes[i, j].imshow(X_train[index], cmap='gray')
        axes[i, j].get_xaxis().set_visible(False)
        axes[i, j].get_yaxis().set_visible(False)
        index += 1
plt.show()

X_train = X_train / 255
X_test = X_test / 255

# Предварительная обработка названий для картинок
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoder.fit(y_train)
y_train = one_hot_encoder.transform(y_train)
y_test = one_hot_encoder.transform(y_test)

# Инициализация CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
input_shape = (X_train.shape[1], X_train.shape[2], 1)

model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D((2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', tensorflow.keras.metrics.Precision(), tensorflow.keras.metrics.Recall()])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[es])

# Результаты обучения модели
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

prediction = model.predict(X_test)
prediction = one_hot_encoder.inverse_transform(prediction)
y_test = one_hot_encoder.inverse_transform(y_test)
cm = confusion_matrix(y_test, prediction)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, cbar=False, xticklabels=labels, yticklabels=labels, fmt='d', annot=True, cmap=plt.cm.Blues)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

predictions_test = np.argmax(model.predict(X_test), axis=1)
print('Accuracy для тестового множества:', metrics.accuracy_score(y_test, predictions_test))
print('Precision для тестового множества:', metrics.precision_score(y_test, predictions_test, average='macro'))
print('Recall для тестового множества:', metrics.recall_score(y_test, predictions_test, average='macro'))
print('F1-measure для тестового множества:', metrics.f1_score(y_test, predictions_test, average='macro'))

predictions_train = np.argmax(model.predict(X_train), axis=1)
y_train = one_hot_encoder.inverse_transform(y_train)
print('Accuracy для тренировочного множества:', metrics.accuracy_score(y_train, predictions_train))
print('Precision для тренировочного множества:', metrics.precision_score(y_train, predictions_train, average='macro'))
print('Recall для тренировочного множества:', metrics.recall_score(y_train, predictions_train, average='macro'))
print('F1-measure для тренировочного множества:', metrics.f1_score(y_train, predictions_train, average='macro'))