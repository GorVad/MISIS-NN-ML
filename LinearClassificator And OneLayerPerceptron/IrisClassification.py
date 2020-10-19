import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression


class Perceptron(object):
    def __init__(self, Learn_Rate=0.5, Iterations=10):
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

    # Изменение весов
    def net_input(self, x):
        return np.dot(x, self.weights[1:]) + self.weights[0]

    # Прогнозирование значения
    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # Создание маркеров для цветов областей классификации
    colors = ('red', 'green', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # Закраска областей
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

# Изначальный разброс тренировочной выборки
yTrain = pd.read_csv("D:\Development\PyCharm\LinearClassification\LinearClassificator And OneLayerPerceptron\DataSets\irisTrainDataSet.csv", header=None)
xTrain = yTrain.iloc[0:75, [0, 2]].values
plt.scatter(xTrain[:25, 0], xTrain[:25, 1], color='red')
plt.scatter(xTrain[25:50, 0], xTrain[25:50, 1], color='blue')
plt.scatter(xTrain[50:75, 0], xTrain[50:75, 1], color='green')
plt.title('Изначальный разброс тренировочной выборки')
plt.xlabel('sepal length in cm')
plt.ylabel('petal length in cm')
plt.show()
# Изначальный разброс тестовой выборки
yTest = pd.read_csv("D:\Development\PyCharm\LinearClassification\LinearClassificator And OneLayerPerceptron\DataSets\irisTestDataSet.csv", header=None)
xTest = yTest.iloc[0:75, [0, 2]].values
plt.scatter(xTest[:25, 0], xTest[:25, 1], color='red')
plt.scatter(xTest[25:50, 0], xTest[25:50, 1], color='blue')
plt.scatter(xTest[50:75, 0], xTest[50:75, 1], color='green')
plt.title('Изначальный разброс тестовой выборки')
plt.xlabel('sepal length in cm')
plt.ylabel('petal length in cm')
plt.show()

Classifier = Perceptron(Learn_Rate=0.01, Iterations=100)
logreg_clf = LogisticRegression()
# Отображение классификации тренировочной выборки по однослойному перцептрону
yTrain = pd.read_csv("D:\Development\PyCharm\LinearClassification\LinearClassificator And OneLayerPerceptron\DataSets\irisTrainDataSet.csv", header=None)
xTrain = yTrain.iloc[0:75, [0, 2]].values
plt.scatter(xTrain[:25, 0], xTrain[:25, 1], color='red')
plt.scatter(xTrain[25:50, 0], xTrain[25:50, 1], color='blue')
plt.scatter(xTrain[50:75, 0], xTrain[50:75, 1], color='green')
yTrain = yTrain.iloc[0:75, 4].values
yTrain = np.where(yTrain=='Iris-setosa', -1, 1)
Classifier.fit(xTrain, yTrain)
plot_decision_regions(xTrain, yTrain, classifier=Classifier)
plt.title('Однослойный перцептрон. Тренировочная выборка')
plt.xlabel('sepal length in cm')
plt.ylabel('petal length in cm')
plt.show()

# Отображение классификации тестовой выборки по однослойному перцептрону
yTest = pd.read_csv("D:\Development\PyCharm\LinearClassification\LinearClassificator And OneLayerPerceptron\DataSets\irisTestDataSet.csv", header=None)
xTest = yTest.iloc[0:75, [0, 2]].values
plt.scatter(xTest[:25, 0], xTest[:25, 1], color='red')
plt.scatter(xTest[25:50, 0], xTest[25:50, 1], color='blue')
plt.scatter(xTest[50:75, 0], xTest[50:75, 1], color='green')
yTest = yTest.iloc[0:75, 4].values
yTest = np.where(yTest=='Iris-setosa', -1, 1)
plot_decision_regions(xTest, yTest, classifier=Classifier)
plt.title('Однослойный перцептрон. Тестовая выборка')
plt.xlabel('sepal length in cm')
plt.ylabel('petal length in cm')
plt.show()

# Отображение классификации тренировочной выборки по линейному классификатору (логистическая регрессия)
yTrain = pd.read_csv("D:\Development\PyCharm\LinearClassification\LinearClassificator And OneLayerPerceptron\DataSets\irisTrainDataSet.csv", header=None)
xTrain = yTrain.iloc[0:75, [0, 2]].values
plt.scatter(xTrain[:25, 0], xTrain[:25, 1], color='red')
plt.scatter(xTrain[25:50, 0], xTrain[25:50, 1], color='blue')
plt.scatter(xTrain[50:75, 0], xTrain[50:75, 1], color='green')
yTrain = yTrain.iloc[0:75, 4].values
yTrain = np.where(yTrain=='Iris-setosa', -1, 1)
logreg_clf.fit(xTrain, yTrain)
plot_decision_regions(xTrain, yTrain, classifier=logreg_clf)
plt.title('Линейный классификатор. Тренировочная выборка')
plt.xlabel('sepal length in cm')
plt.ylabel('petal length in cm')
plt.show()

# Отображение классификации тестовой выборки по линейному классификатору (логистическая регрессия)
yTest = pd.read_csv("D:\Development\PyCharm\LinearClassification\LinearClassificator And OneLayerPerceptron\DataSets\irisTestDataSet.csv", header=None)
xTest = yTest.iloc[0:75, [0, 2]].values
plt.scatter(xTest[:25, 0], xTest[:25, 1], color='red')
plt.scatter(xTest[25:50, 0], xTest[25:50, 1], color='blue')
plt.scatter(xTest[50:75, 0], xTest[50:75, 1], color='green')
yTest = yTest.iloc[0:75, 4].values
yTest = np.where(yTest=='Iris-setosa', -1, 1)
plot_decision_regions(xTest, yTest, classifier=logreg_clf)
plt.title('Линейный классификатор. Тестовая выборка')
plt.xlabel('sepal length in cm')
plt.ylabel('petal length in cm')
plt.show()
