# Importing dependencies.
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap


# Creation of the main perceptron object.
class Perceptron(object):
    # Initiating the learning rate and number of iterations.
    def __init__(self, Learn_Rate=0.7, Iterations=30):
        self.learn_rate = Learn_Rate
        self.Iterations = Iterations
        self.errors = []
        self.weights = np.zeros(1 + xTrain.shape[1])

    # Defining fit method for model training.
    def fit(self, x, y):
        self.weights = np.zeros(1 + x.shape[1])
        for i in range(self.Iterations):
            error = 0
            for xi, target in zip(x, y):
                update = self.learn_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                error += int(update != 0)
            self.errors.append(error)
        return self

    # Net Input method for summing the given matrix inputs and their corresponding weights.
    def net_input(self, x):
        return np.dot(x, self.weights[1:]) + self.weights[0]

    # Predict method for predicting the classification of data inputs.
    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


# Data retrieval and preperation. Training Data
yTrain = pd.read_csv("1. LinearClassificator And OneLayerPerceptron/DataSets/wineTrainDataSet.csv", header=None)
xTrain = yTrain.iloc[0:92, [7, 12]].values
plt.scatter(xTrain[:35, 0], xTrain[:35, 1], color='red')
plt.scatter(xTrain[35:69, 0], xTrain[35:69, 1], color='blue')
plt.scatter(xTrain[69:92, 0], xTrain[69:92, 1], color='yellow')
plt.title('Изначальный разброс тренировочной выборки')
plt.show()
yTrain = yTrain.iloc[0:92, 0].values
yTrain = np.where(yTrain==3, -1, 1)

# Data retrieval and preperation. Prediction Data
yPredict = pd.read_csv("1. LinearClassificator And OneLayerPerceptron/DataSets/winePredictDataSet.csv", header=None)
xPredict = yPredict.iloc[0:86, [7, 12]].values
plt.scatter(xPredict[:24, 0], xPredict[:24, 1], color='red')
plt.scatter(xPredict[24:62, 0], xPredict[24:62, 1], color='blue')
plt.scatter(xPredict[62:86, 0], xPredict[62:86, 1], color='yellow')
plt.title('Изначальный разброс тестовой выборки')
plt.show()
yPredict = yPredict.iloc[0:86, 0].values
yPredict = np.where(yPredict==3, -1, 1)

# Model training and evaluation.
Classifier = Perceptron(Learn_Rate=0.01, Iterations=100)
Classifier.fit(xTrain, yTrain)
plt.plot(range(1, len(Classifier.errors) + 1), Classifier.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.title('Модель обучения')
plt.show()


# Defining function that plots the decision regions.
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 'o', 'o', 'o', 'v')
    colors = ('red', 'green', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


# Showing the final results of the perceptron model.Training Data
plot_decision_regions(xTrain, yTrain, classifier=Classifier)
plt.title('Тренировочная выборка')
plt.show()
# Showing the final results of the perceptron model.Prediction Data
plot_decision_regions(xPredict, yPredict, classifier=Classifier)
plt.title('Тестовая выборка')
plt.show()