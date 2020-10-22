import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import metrics


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


irisTrainDataSet = pd.read_csv("D:\Development\PyCharm\LinearClassification\LinearClassificator And OneLayerPerceptron\DataSets\irisTrainDataSet.csv", header=None)
irisTestDataSet = pd.read_csv("D:\Development\PyCharm\LinearClassification\LinearClassificator And OneLayerPerceptron\DataSets\irisTestDataSet.csv", header=None)
# Изначальный разброс тренировочной выборки
yTrain = irisTrainDataSet
xTrain = yTrain.iloc[0:75, [0, 2]].values
plt.scatter(xTrain[:25, 0], xTrain[:25, 1], color='red')
plt.scatter(xTrain[25:50, 0], xTrain[25:50, 1], color='blue')
plt.scatter(xTrain[50:75, 0], xTrain[50:75, 1], color='green')
plt.title('Изначальный разброс тренировочной выборки')
plt.xlabel('sepal length in cm')
plt.ylabel('petal length in cm')
plt.show()
# Изначальный разброс тестовой выборки
yTest = irisTestDataSet
xTest = yTest.iloc[0:75, [0, 2]].values
plt.scatter(xTest[:25, 0], xTest[:25, 1], color='red')
plt.scatter(xTest[25:50, 0], xTest[25:50, 1], color='blue')
plt.scatter(xTest[50:75, 0], xTest[50:75, 1], color='green')
plt.title('Изначальный разброс тестовой выборки')
plt.xlabel('sepal length in cm')
plt.ylabel('petal length in cm')
plt.show()

Classifier_Setosa = Perceptron()
Classifier_Virginica = Perceptron()
Classifier_Versicolor = Perceptron()
logreg_clf_Setosa = LogisticRegression()
logreg_clf_Virginica = LogisticRegression()
logreg_clf_Versicolor = LogisticRegression()
# Отображение классификации тренировочной выборки по однослойному перцептрону
yTrain_Setosa = irisTrainDataSet
xTrain_Setosa = yTrain_Setosa.iloc[0:75, [0, 2]].values
yTrain_Setosa = yTrain_Setosa.iloc[0:75, 4].values
yTrain_Setosa = np.where(yTrain_Setosa=='Iris-setosa', -1, 1)
Classifier_Setosa.fit(xTrain_Setosa, yTrain_Setosa)
plot_decision_regions(xTrain_Setosa, yTrain_Setosa, classifier=Classifier_Setosa)
yTrain_virginica = irisTrainDataSet
xTrain_virginica = yTrain_virginica.iloc[0:75, [0, 2]].values
yTrain_virginica = yTrain_virginica.iloc[0:75, 4].values
yTrain_virginica = np.where(yTrain_virginica=='Iris-virginica', -1, 1)
Classifier_Virginica.fit(xTrain_virginica, yTrain_virginica)
plot_decision_regions(xTrain_virginica, yTrain_virginica, classifier=Classifier_Virginica)
yTrain_versicolor = irisTrainDataSet
xTrain_versicolor = yTrain_versicolor.iloc[0:75, [0, 2]].values
yTrain_versicolor = yTrain_versicolor.iloc[0:75, 4].values
yTrain_versicolor = np.where(yTrain_versicolor=='Iris-versicolor', -1, 1)
Classifier_Versicolor.fit(xTrain_versicolor, yTrain_versicolor)
plt.scatter(xTrain_Setosa[:25, 0], xTrain_Setosa[:25, 1], color='red')
plt.scatter(xTrain_Setosa[25:50, 0], xTrain_Setosa[25:50, 1], color='blue')
plt.scatter(xTrain_Setosa[50:75, 0], xTrain_Setosa[50:75, 1], color='green')
plt.title('Однослойный перцептрон. Тренировочная выборка')
plt.xlabel('sepal length in cm')
plt.ylabel('petal length in cm')
plt.show()

# Отображение классификации тестовой выборки по однослойному перцептрону
yTest_Setosa = irisTestDataSet
xTest_Setosa = yTest_Setosa.iloc[0:75, [0, 2]].values
yTest_Setosa = yTest_Setosa.iloc[0:75, 4].values
yTest_Setosa = np.where(yTest_Setosa=='Iris-setosa', -1, 1)
plot_decision_regions(xTest_Setosa, yTest_Setosa, classifier=Classifier_Setosa)
yTest_virginica = irisTestDataSet
xTest_virginica = yTest_virginica.iloc[0:75, [0, 2]].values
yTest_virginica = yTest_virginica.iloc[0:75, 4].values
yTest_virginica = np.where(yTest_virginica=='Iris-virginica', -1, 1)
plot_decision_regions(xTest_virginica, yTest_virginica, classifier=Classifier_Virginica)
yTest_versicolor = irisTestDataSet
xTest_versicolor = yTest_versicolor.iloc[0:75, [0, 2]].values
yTest_versicolor = yTest_versicolor.iloc[0:75, 4].values
yTest_versicolor = np.where(yTest_versicolor=='Iris-versicolor', -1, 1)
plt.scatter(xTest_Setosa[:25, 0], xTest_Setosa[:25, 1], color='red')
plt.scatter(xTest_Setosa[25:50, 0], xTest_Setosa[25:50, 1], color='blue')
plt.scatter(xTest_Setosa[50:75, 0], xTest_Setosa[50:75, 1], color='green')
plt.title('Однослойный перцептрон. Тестовая выборка')
plt.xlabel('sepal length in cm')
plt.ylabel('petal length in cm')
plt.show()

# Рассчет метрик качества для Однослойного перцептрона
print('Accuracy for Iris Single layer Perceptron Setosa:',metrics.accuracy_score(yTest_Setosa, Classifier_Setosa.predict(xTest_Setosa)))
print('Precision for Iris Single layer Perceptron Setosa:',metrics.precision_score(yTest_Setosa, Classifier_Setosa.predict(xTest_Setosa)))
print('Recall for Iris Single layer Perceptron Setosa:',metrics.recall_score(yTest_Setosa, Classifier_Setosa.predict(xTest_Setosa)))
print('F1-measure for Iris Single layer Perceptron Setosa:',metrics.f1_score(yTest_Setosa, Classifier_Setosa.predict(xTest_Setosa)))
print('Accuracy for Iris Single layer Perceptron Virginica:',metrics.accuracy_score(yTest_virginica, Classifier_Virginica.predict(xTest_virginica)))
print('Precision for Iris Single layer Perceptron Virginica:',metrics.precision_score(yTest_virginica, Classifier_Virginica.predict(xTest_virginica)))
print('Recall for Iris Single layer Perceptron Virginica:',metrics.recall_score(yTest_virginica, Classifier_Virginica.predict(xTest_virginica)))
print('F1-measure for Iris Single layer Perceptron Virginica:',metrics.f1_score(yTest_virginica, Classifier_Virginica.predict(xTest_virginica)))
print('Accuracy for Iris Single layer Perceptron Versicolor:',metrics.accuracy_score(yTest_versicolor, Classifier_Versicolor.predict(xTest_versicolor)))
print('Precision for Iris Single layer Perceptron Versicolor:',metrics.precision_score(yTest_versicolor, Classifier_Versicolor.predict(xTest_versicolor)))
print('Recall for Iris Single layer Perceptron Versicolor:',metrics.recall_score(yTest_versicolor, Classifier_Versicolor.predict(xTest_versicolor)))
print('F1-measure for Iris Single layer Perceptron Versicolor:',metrics.f1_score(yTest_versicolor, Classifier_Versicolor.predict(xTest_versicolor)))


# Отображение классификации тренировочной выборки по линейному классификатору (логистическая регрессия)
yTrain_Setosa = irisTrainDataSet
xTrain_Setosa = yTrain_Setosa.iloc[0:75, [0, 2]].values
yTrain_Setosa = yTrain.iloc[0:75, 4].values
yTrain_Setosa = np.where(yTrain_Setosa=='Iris-setosa', -1, 1)
logreg_clf_Setosa.fit(xTrain_Setosa, yTrain_Setosa)
plot_decision_regions(xTrain_Setosa, yTrain_Setosa, classifier=logreg_clf_Setosa)
yTrain_virginica = irisTrainDataSet
xTrain_virginica = yTrain_virginica.iloc[0:75, [0, 2]].values
yTrain_virginica = yTrain_virginica.iloc[0:75, 4].values
yTrain_virginica = np.where(yTrain_virginica=='Iris-virginica', -1, 1)
logreg_clf_Virginica.fit(xTrain_virginica, yTrain_virginica)
plot_decision_regions(xTrain_virginica, yTrain_virginica, classifier=logreg_clf_Virginica)
yTrain_versicolor = irisTrainDataSet
xTrain_versicolor = yTrain_versicolor.iloc[0:75, [0, 2]].values
yTrain_versicolor = yTrain_versicolor.iloc[0:75, 4].values
yTrain_versicolor = np.where(yTrain_versicolor=='Iris-versicolor', -1, 1)
logreg_clf_Versicolor.fit(xTrain_versicolor, yTrain_versicolor)
plt.scatter(xTrain_Setosa[:25, 0], xTrain_Setosa[:25, 1], color='red')
plt.scatter(xTrain_Setosa[25:50, 0], xTrain_Setosa[25:50, 1], color='blue')
plt.scatter(xTrain_Setosa[50:75, 0], xTrain_Setosa[50:75, 1], color='green')
plt.title('Линейный классификатор. Тренировочная выборка')
plt.xlabel('sepal length in cm')
plt.ylabel('petal length in cm')
plt.show()

# Отображение классификации тестовой выборки по линейному классификатору (логистическая регрессия)
yTest_Setosa = irisTestDataSet
xTest_Setosa = yTest_Setosa.iloc[0:75, [0, 2]].values
yTest_Setosa = yTest_Setosa.iloc[0:75, 4].values
yTest_Setosa = np.where(yTest_Setosa=='Iris-setosa', -1, 1)
plot_decision_regions(xTest_Setosa, yTest_Setosa, classifier=logreg_clf_Setosa)
yTest_virginica = irisTestDataSet
xTest_virginica = yTest_virginica.iloc[0:75, [0, 2]].values
yTest_virginica = yTest_virginica.iloc[0:75, 4].values
yTest_virginica = np.where(yTest_virginica=='Iris-virginica', -1, 1)
plot_decision_regions(xTest_virginica, yTest_virginica, classifier=logreg_clf_Virginica)
yTest_versicolor = irisTestDataSet
xTest_versicolor = yTest_versicolor.iloc[0:75, [0, 2]].values
yTest_versicolor = yTest_versicolor.iloc[0:75, 4].values
yTest_versicolor = np.where(yTest_versicolor=='Iris-versicolor', -1, 1)
plt.scatter(xTest_Setosa[:25, 0], xTest_Setosa[:25, 1], color='red')
plt.scatter(xTest_Setosa[25:50, 0], xTest_Setosa[25:50, 1], color='blue')
plt.scatter(xTest_Setosa[50:75, 0], xTest_Setosa[50:75, 1], color='green')
plt.title('Линейный классификатор. Тестовая выборка')
plt.xlabel('sepal length in cm')
plt.ylabel('petal length in cm')
plt.show()

# Рассчет метрик качества для Линейного классификатора
print('Accuracy for Iris Linear Classification Setosa:',metrics.accuracy_score(yTest_Setosa, logreg_clf_Setosa.predict(xTest_Setosa)))
print('Precision for Iris Linear Classification Setosa:',metrics.precision_score(yTest_Setosa, logreg_clf_Setosa.predict(xTest_Setosa)))
print('Recall for Iris Linear Classification Setosa:',metrics.recall_score(yTest_Setosa, logreg_clf_Setosa.predict(xTest_Setosa)))
print('F1-measure for Iris Linear Classification Setosa:',metrics.f1_score(yTest_Setosa, logreg_clf_Setosa.predict(xTest_Setosa)))
print('Accuracy for Iris Linear Classification Virginica:',metrics.accuracy_score(yTest_virginica, logreg_clf_Virginica.predict(xTest_virginica)))
print('Precision for Iris Linear Classification Virginica:',metrics.precision_score(yTest_virginica, logreg_clf_Virginica.predict(xTest_virginica)))
print('Recall for Iris Linear Classification Virginica:',metrics.recall_score(yTest_virginica, logreg_clf_Virginica.predict(xTest_virginica)))
print('F1-measure for Iris Linear Classification Virginica:',metrics.f1_score(yTest_virginica, logreg_clf_Virginica.predict(xTest_virginica)))
print('Accuracy for Iris Linear Classification Versicolor:',metrics.accuracy_score(yTest_versicolor, logreg_clf_Versicolor.predict(xTest_versicolor)))
print('Precision for Iris Linear Classification Versicolor:',metrics.precision_score(yTest_versicolor, logreg_clf_Versicolor.predict(xTest_versicolor)))
print('Recall for Iris Linear Classification Versicolor:',metrics.recall_score(yTest_versicolor, logreg_clf_Versicolor.predict(xTest_versicolor)))
print('F1-measure for Iris Linear Classification Versicolor:',metrics.f1_score(yTest_versicolor, logreg_clf_Versicolor.predict(xTest_versicolor)))
