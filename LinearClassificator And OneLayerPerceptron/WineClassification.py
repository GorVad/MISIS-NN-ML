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

wineTrainDataSet = pd.read_csv("D:\Development\PyCharm\LinearClassification\LinearClassificator And OneLayerPerceptron\DataSets\wineTrainDataSet.csv", header=None)
wineTestDataSet = pd.read_csv("D:\Development\PyCharm\LinearClassification\LinearClassificator And OneLayerPerceptron\DataSets\wineTestDataSet.csv", header=None)
# Изначальный разброс тренировочной выборки
yTrain = wineTrainDataSet
xTrain = yTrain.iloc[0:92, [7, 12]].values
plt.scatter(xTrain[:35, 0], xTrain[:35, 1], color='red')
plt.scatter(xTrain[35:69, 0], xTrain[35:69, 1], color='blue')
plt.scatter(xTrain[69:92, 0], xTrain[69:92, 1], color='green')
plt.title('Изначальный разброс тренировочной выборки')
plt.xlabel('Total phenols')
plt.ylabel('Hue')
plt.show()
# Изначальный разброс тестовой выборки
yTest = wineTestDataSet
xTest = yTest.iloc[0:86, [7, 12]].values
plt.scatter(xTest[:24, 0], xTest[:24, 1], color='blue')
plt.scatter(xTest[24:62, 0], xTest[24:62, 1], color='green')
plt.scatter(xTest[62:86, 0], xTest[62:86, 1], color='red')
plt.title('Изначальный разброс тестовой выборки')
plt.xlabel('Total phenols')
plt.ylabel('Hue')
plt.show()

Classifier_Class1 = Perceptron()
Classifier_Class2 = Perceptron()
Classifier_Class3 = Perceptron()
logreg_clf_Class1 = LogisticRegression()
logreg_clf_Class2 = LogisticRegression()
logreg_clf_Class3 = LogisticRegression()

# Отображение классификации тренировочной выборки по однослойному перцептрону
yTrain_class1 = wineTrainDataSet
xTrain_class1 = yTrain_class1.iloc[0:92, [7, 12]].values
yTrain_class1 = yTrain_class1.iloc[0:92, 0].values
yTrain_class1 = np.where(yTrain_class1==3, -1, 1)
Classifier_Class1.fit(xTrain_class1, yTrain_class1)
plot_decision_regions(xTrain_class1, yTrain_class1, classifier=Classifier_Class1)
yTrain_Class2 = wineTrainDataSet
xTrain_Class2 = yTrain_Class2.iloc[0:92, [7, 12]].values
yTrain_Class2 = yTrain_Class2.iloc[0:92, 0].values
yTrain_Class2 = np.where(yTrain_Class2==1, -1, 1)
Classifier_Class2.fit(xTrain_Class2, yTrain_Class2)
plot_decision_regions(xTrain_Class2, yTrain_Class2, classifier=Classifier_Class2)
yTrain_class3 = wineTrainDataSet
xTrain_class3 = yTrain_class3.iloc[0:92, [7, 12]].values
yTrain_class3 = yTrain_class3.iloc[0:92, 0].values
yTrain_class3 = np.where(yTrain_class3==2, -1, 1)
Classifier_Class3.fit(xTrain_class3, yTrain_class3)
plt.scatter(xTrain_class1[:35, 0], xTrain_class1[:35, 1], color='blue')
plt.scatter(xTrain_class1[35:69, 0], xTrain_class1[35:69, 1], color='green')
plt.scatter(xTrain_class1[69:92, 0], xTrain_class1[69:92, 1], color='red')
plt.title('Однослойный перцептрон. Тренировочная выборка')
plt.xlabel('Total phenols')
plt.ylabel('Hue')
plt.show()

# Отображение классификации тестовой выборки по однослойному перцептрону
yTest_class1 = wineTestDataSet
xTest_class1 = yTest_class1.iloc[0:86, [7, 12]].values
yTest_class1 = yTest_class1.iloc[0:86, 0].values
yTest_class1 = np.where(yTest_class1==3, -1, 1)
plot_decision_regions(xTest_class1, yTest_class1, classifier=Classifier_Class1)
yTest_class2 = wineTestDataSet
xTest_class2 = yTest_class2.iloc[0:86, [7, 12]].values
yTest_class2 = yTest_class2.iloc[0:86, 0].values
yTest_class2 = np.where(yTest_class2==1, -1, 1)
plot_decision_regions(xTest_class2, yTest_class2, classifier=Classifier_Class2)
yTest_class3 = wineTestDataSet
xTest_class3 = yTest_class3.iloc[0:86, [7, 12]].values
yTest_class3= yTest_class3.iloc[0:86, 0].values
yTest_class3 = np.where(yTest_class3==2, -1, 1)
plt.scatter(xTest_class1[:24, 0], xTest_class1[:24, 1], color='blue')
plt.scatter(xTest_class1[24:62, 0], xTest_class1[24:62, 1], color='green')
plt.scatter(xTest_class1[62:86, 0], xTest_class1[62:86, 1], color='red')
plt.title('Однослойный перцептрон. Тестовая выборка')
plt.xlabel('Total phenols')
plt.ylabel('Hue')
plt.show()

# Рассчет метрик качества для Однослойного перцептрона
print('Accuracy for Iris Single layer Perceptron Class1:',metrics.accuracy_score(yTest_class1, Classifier_Class1.predict(xTest_class1)))
print('Precision for Iris Single layer Perceptron Class1:',metrics.precision_score(yTest_class1, Classifier_Class1.predict(xTest_class1)))
print('Recall for Iris Single layer Perceptron Class1:',metrics.recall_score(yTest_class1, Classifier_Class1.predict(xTest_class1)))
print('F1-measure for Iris Single layer Perceptron Class1:',metrics.f1_score(yTest_class1, Classifier_Class1.predict(xTest_class1)))
print('Accuracy for Iris Single layer Perceptro Class2:',metrics.accuracy_score(yTest_class2, Classifier_Class2.predict(xTest_class2)))
print('Precision for Iris Single layer Perceptron Class2:',metrics.precision_score(yTest_class2, Classifier_Class2.predict(xTest_class2)))
print('Recall for Iris Single layer Perceptron Class2:',metrics.recall_score(yTest_class2, Classifier_Class2.predict(xTest_class2)))
print('F1-measure for Iris Single layer Perceptron Class2:',metrics.f1_score(yTest_class2, Classifier_Class2.predict(xTest_class2)))
print('Accuracy for Iris Single layer Perceptron Class3:',metrics.accuracy_score(yTest_class3, Classifier_Class3.predict(xTest_class3)))
print('Precision for Iris Single layer Perceptron Class3:',metrics.precision_score(yTest_class3, Classifier_Class3.predict(xTest_class3)))
print('Recall for Iris Single layer Perceptron Class3:',metrics.recall_score(yTest_class3, Classifier_Class3.predict(xTest_class3)))
print('F1-measure for Iris Single layer Perceptron Class3:',metrics.f1_score(yTest_class3, Classifier_Class3.predict(xTest_class3)))


# Отображение классификации тренировочной выборки по линейному классификатору (логистическая регрессия)
yTrain_class1 = wineTrainDataSet
xTrain_class1 = yTrain_class1.iloc[0:92, [7, 12]].values
yTrain_class1 = yTrain_class1.iloc[0:92, 0].values
yTrain_class1 = np.where(yTrain_class1==3, -1, 1)
logreg_clf_Class1.fit(xTrain_class1,yTrain_class1)
plot_decision_regions(xTrain_class1, yTrain_class1, classifier=logreg_clf_Class1)
yTrain_Class2 = wineTrainDataSet
xTrain_Class2 = yTrain_Class2.iloc[0:92, [7, 12]].values
yTrain_Class2 = yTrain_Class2.iloc[0:92, 0].values
yTrain_Class2 = np.where(yTrain_Class2==1, -1, 1)
logreg_clf_Class2.fit(xTrain_Class2,yTrain_Class2)
plot_decision_regions(xTrain_Class2, yTrain_Class2, classifier=logreg_clf_Class2)
yTrain_class3 = wineTrainDataSet
xTrain_class3 = yTrain_class3.iloc[0:92, [7, 12]].values
yTrain_class3 = yTrain_class3.iloc[0:92, 0].values
yTrain_class3 = np.where(yTrain_class3==2, -1, 1)
logreg_clf_Class3.fit(xTrain_class3,yTrain_class3)
plt.scatter(xTrain_class1[:35, 0], xTrain_class1[:35, 1], color='blue')
plt.scatter(xTrain_class1[35:69, 0], xTrain_class1[35:69, 1], color='green')
plt.scatter(xTrain_class1[69:92, 0], xTrain_class1[69:92, 1], color='red')
plt.title('Линейный классификатор. Тренировочная выборка')
plt.xlabel('Total phenols')
plt.ylabel('Hue')
plt.show()

# Отображение классификации тестовой выборки по линейному классификатору (логистическая регрессия)
yTest_class1 = wineTestDataSet
xTest_class1 = yTest_class1.iloc[0:86, [7, 12]].values
yTest_class1 = yTest_class1.iloc[0:86, 0].values
yTest_class1 = np.where(yTest_class1==3, -1, 1)
plot_decision_regions(xTest, yTest, classifier=logreg_clf_Class1)
yTest_class2 = wineTestDataSet
xTest_class2 = yTest_class2.iloc[0:86, [7, 12]].values
yTest_class2 = yTest_class2.iloc[0:86, 0].values
yTest_class2 = np.where(yTest_class2==1, -1, 1)
plot_decision_regions(xTest_class2, yTest_class2, classifier=logreg_clf_Class2)
yTest_class3 = wineTestDataSet
xTest_class3 = yTest_class3.iloc[0:86, [7, 12]].values
yTest_class3 = yTest_class3.iloc[0:86, 0].values
yTest_class3 = np.where(yTest_class3==2, -1, 1)
plt.scatter(xTest_class1[:24, 0], xTest_class1[:24, 1], color='blue')
plt.scatter(xTest[24:62, 0], xTest_class1[24:62, 1], color='green')
plt.scatter(xTest_class1[62:86, 0], xTest_class1[62:86, 1], color='red')
plt.title('Линейный классификатор. Тестовая выборка')
plt.xlabel('Total phenols')
plt.ylabel('Hue')
plt.show()

# Рассчет метрик качества для Линейного классификатора
print('Accuracy for Iris Linear Classification Class1:',metrics.accuracy_score(yTest_class1, logreg_clf_Class1.predict(xTest_class1)))
print('Precision for Iris Linear Classification Class1:',metrics.precision_score(yTest_class1, logreg_clf_Class1.predict(xTest_class1)))
print('Recall for Iris Linear Classification Class1:',metrics.recall_score(yTest_class1, logreg_clf_Class1.predict(xTest_class1)))
print('F1-measure for Iris Linear Classification Class1:',metrics.f1_score(yTest_class1, logreg_clf_Class1.predict(xTest_class1)))
print('Accuracy for Iris Linear Classification Class2:',metrics.accuracy_score(yTest_class2, logreg_clf_Class2.predict(xTest_class2)))
print('Precision for Iris Linear Classification Class2:',metrics.precision_score(yTest_class2, logreg_clf_Class2.predict(xTest_class2)))
print('Recall for Iris Linear Classification Class2:',metrics.recall_score(yTest_class2, logreg_clf_Class2.predict(xTest_class2)))
print('F1-measure for Iris Linear Classification Class2:',metrics.f1_score(yTest_class2, logreg_clf_Class2.predict(xTest_class2)))
print('Accuracy for Iris Linear Classification Class3:',metrics.accuracy_score(yTest_class3, logreg_clf_Class3.predict(xTest_class3)))
print('Precision for Iris Linear Classification Class3:',metrics.precision_score(yTest_class3, logreg_clf_Class3.predict(xTest_class3)))
print('Recall for Iris Linear Classification Class3:',metrics.recall_score(yTest_class3, logreg_clf_Class3.predict(xTest_class3)))
print('F1-measure for Iris Linear Classification Class3:',metrics.f1_score(yTest_class3, logreg_clf_Class3.predict(xTest_class3)))
