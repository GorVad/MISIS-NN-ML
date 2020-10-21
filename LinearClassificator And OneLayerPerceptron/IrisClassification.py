import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.neural_network import MLPClassifier


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

Classifier = Perceptron()
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

# Рассчет метрик качества для Однослойного перцептрона
print('Accuracy for Iris Single layer Perceptron:',metrics.accuracy_score(yTest, Classifier.predict(xTest)))
print('Precision for Iris Single layer Perceptron:',metrics.precision_score(yTest, Classifier.predict(xTest)))
print('Recall for Iris Single layer Perceptron:',metrics.recall_score(yTest, Classifier.predict(xTest)))
print('F1-measure for Iris Single layer Perceptron:',metrics.f1_score(yTest, Classifier.predict(xTest)))


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

# Рассчет метрик качества для Линейного классификатора
print('Accuracy for Iris Linear Classification :',metrics.accuracy_score(yTest, logreg_clf.predict(xTest)))
print('Precision for Iris Linear Classification:',metrics.precision_score(yTest, logreg_clf.predict(xTest)))
print('Recall for Iris Linear Classification:',metrics.recall_score(yTest, logreg_clf.predict(xTest)))
print('F1-measure for Iris Linear Classification:',metrics.f1_score(yTest, logreg_clf.predict(xTest)))