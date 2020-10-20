import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


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
yTrain = pd.read_csv("D:\Development\PyCharm\LinearClassification\LinearClassificator And OneLayerPerceptron\DataSets\wineTrainDataSet.csv", header=None)
xTrain = yTrain.iloc[0:92, [7, 12]].values
plt.scatter(xTrain[:35, 0], xTrain[:35, 1], color='red')
plt.scatter(xTrain[35:69, 0], xTrain[35:69, 1], color='blue')
plt.scatter(xTrain[69:92, 0], xTrain[69:92, 1], color='green')
plt.title('Изначальный разброс тренировочной выборки')
plt.xlabel('Total phenols')
plt.ylabel('Hue')
plt.show()
# Изначальный разброс тестовой выборки
yTest = pd.read_csv("D:\Development\PyCharm\LinearClassification\LinearClassificator And OneLayerPerceptron\DataSets\wineTestDataSet.csv", header=None)
xTest = yTest.iloc[0:86, [7, 12]].values
plt.scatter(xTest[:24, 0], xTest[:24, 1], color='blue')
plt.scatter(xTest[24:62, 0], xTest[24:62, 1], color='green')
plt.scatter(xTest[62:86, 0], xTest[62:86, 1], color='red')
plt.title('Изначальный разброс тестовой выборки')
plt.xlabel('Total phenols')
plt.ylabel('Hue')
plt.show()

Classifier = Perceptron(Learn_Rate=0.01, Iterations=100)
logreg_clf = LogisticRegression()
# Отображение классификации тренировочной выборки по однослойному перцептрону
yTrain = pd.read_csv("D:\Development\PyCharm\LinearClassification\LinearClassificator And OneLayerPerceptron\DataSets\wineTrainDataSet.csv", header=None)
xTrain = yTrain.iloc[0:92, [7, 12]].values
plt.scatter(xTrain[:35, 0], xTrain[:35, 1], color='blue')
plt.scatter(xTrain[35:69, 0], xTrain[35:69, 1], color='green')
plt.scatter(xTrain[69:92, 0], xTrain[69:92, 1], color='red')
yTrain = yTrain.iloc[0:92, 0].values
yTrain = np.where(yTrain==3, -1, 1)
Classifier.fit(xTrain, yTrain)
plot_decision_regions(xTrain, yTrain, classifier=Classifier)
plt.title('Однослойный перцептрон. Тренировочная выборка')
plt.xlabel('Total phenols')
plt.ylabel('Hue')
plt.show()

# Отображение классификации тестовой выборки по однослойному перцептрону
yTest = pd.read_csv("D:\Development\PyCharm\LinearClassification\LinearClassificator And OneLayerPerceptron\DataSets\wineTestDataSet.csv", header=None)
xTest = yTest.iloc[0:86, [7, 12]].values
plt.scatter(xTest[:24, 0], xTest[:24, 1], color='blue')
plt.scatter(xTest[24:62, 0], xTest[24:62, 1], color='green')
plt.scatter(xTest[62:86, 0], xTest[62:86, 1], color='red')
yTest = yTest.iloc[0:86, 0].values
yTest = np.where(yTest==3, -1, 1)
plot_decision_regions(xTest, yTest, classifier=Classifier)
plt.title('Однослойный перцептрон. Тестовая выборка')
plt.xlabel('Total phenols')
plt.ylabel('Hue')
plt.show()

# Отображение классификации тренировочной выборки по линейному классификатору (логистическая регрессия)
yTrain = pd.read_csv("D:\Development\PyCharm\LinearClassification\LinearClassificator And OneLayerPerceptron\DataSets\wineTrainDataSet.csv", header=None)
xTrain = yTrain.iloc[0:92, [7, 12]].values
plt.scatter(xTrain[:35, 0], xTrain[:35, 1], color='blue')
plt.scatter(xTrain[35:69, 0], xTrain[35:69, 1], color='green')
plt.scatter(xTrain[69:92, 0], xTrain[69:92, 1], color='red')
yTrain = yTrain.iloc[0:92, 0].values
yTrain = np.where(yTrain==3, -1, 1)
logreg_clf.fit(xTrain,yTrain)
plot_decision_regions(xTrain, yTrain, classifier=logreg_clf)
plt.title('Линейный классификатор. Тренировочная выборка')
plt.xlabel('Total phenols')
plt.ylabel('Hue')
plt.show()

# Рассчет метрик качества для Однослойного перцептрона
print('Accuracy for Iris Single layer Perceptron:',metrics.accuracy_score(yTest, Classifier.predict(xTest)))
print('Precision for Iris Single layer Perceptron:',metrics.precision_score(yTest, Classifier.predict(xTest)))
print('Recall for Iris Single layer Perceptron:',metrics.recall_score(yTest, Classifier.predict(xTest)))
print('F1-measure for Iris Single layer Perceptron:',metrics.f1_score(yTest, Classifier.predict(xTest)))

# Отображение классификации тестовой выборки по линейному классификатору (логистическая регрессия)
yTest = pd.read_csv("D:\Development\PyCharm\LinearClassification\LinearClassificator And OneLayerPerceptron\DataSets\wineTestDataSet.csv", header=None)
xTest = yTest.iloc[0:86, [7, 12]].values
plt.scatter(xTest[:24, 0], xTest[:24, 1], color='blue')
plt.scatter(xTest[24:62, 0], xTest[24:62, 1], color='green')
plt.scatter(xTest[62:86, 0], xTest[62:86, 1], color='red')
yTest = yTest.iloc[0:86, 0].values
yTest = np.where(yTest==3, -1, 1)
plot_decision_regions(xTest, yTest, classifier=logreg_clf)
plt.title('Линейный классификатор. Тестовая выборка')
plt.xlabel('Total phenols')
plt.ylabel('Hue')
plt.show()

# Рассчет метрик качества для Линейного классификатора
print('Accuracy for Iris Linear Classification :',metrics.accuracy_score(yTest, logreg_clf.predict(xTest)))
print('Precision for Iris Linear Classification:',metrics.precision_score(yTest, logreg_clf.predict(xTest)))
print('Recall for Iris Linear Classification:',metrics.recall_score(yTest, logreg_clf.predict(xTest)))
print('F1-measure for Iris Linear Classification:',metrics.f1_score(yTest, logreg_clf.predict(xTest)))
