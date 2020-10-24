from sklearn import metrics
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

irisDataSet = pd.read_csv("D:\PyCharm\MISIS-NN-ML\LinearClassificator And OneLayerPerceptron\DataSets\iris.csv", header=None)
x = irisDataSet.iloc[:, 1:3].values
y = irisDataSet.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

i = 0
while i<len(y_train):
    if y_train[i] == 1:
        plt.scatter(X_train[i, 0], X_train[i, 1], color='red')
    if y_train[i] == 2:
        plt.scatter(X_train[i, 0], X_train[i, 1], color='green')
    if y_train[i] == 3:
        plt.scatter(X_train[i, 0], X_train[i, 1], color='blue')
    i = i+1
plt.title('Исходная классификация тренировочной выборки')
plt.show()
i = 0
while i<len(y_test):
    if y_test[i] == 1:
        plt.scatter(X_test[i, 0], X_test[i, 1], color='red')
    if y_test[i] == 2:
        plt.scatter(X_test[i, 0], X_test[i, 1], color='green')
    if y_test[i] == 3:
        plt.scatter(X_test[i, 0], X_test[i, 1], color='blue')
    i = i+1
plt.title('Исходная классификация тестовой выборки')
plt.show()

XTrain_std = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
XTest_std = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

singleLayerPerceptron = Perceptron()
singleLayerPerceptron.fit(X_train, y_train)
logisticRegression = LogisticRegression()
logisticRegression.fit(X_test, y_test)

fig = plot_decision_regions(X=XTrain_std, y=y_train, clf=singleLayerPerceptron, legend=2)
plt.title('Однойслойный перцептрон. Тренировочная выборка')
plt.show()
fig = plot_decision_regions(X=XTest_std, y=y_test, clf=singleLayerPerceptron, legend=2)
plt.title('Однойслойный перцептрон. Тестовая выборка')
plt.show()
fig = plot_decision_regions(X=X_train, y=y_train, clf=logisticRegression, legend=2)
plt.title('Логистическая регрессия. Тренировочная выборка')
plt.show()
fig = plot_decision_regions(X=X_test, y=y_test, clf=logisticRegression, legend=2)
plt.title('Логистическая регрессия. Тестовая выборка')
plt.show()

# Рассчет метрик качества для Однослойного перцептрона
print('Accuracy for Single layer Perceptron:',metrics.accuracy_score(y_test, singleLayerPerceptron.predict(X_test)))
print('Precision for Single layer Perceptron:',metrics.precision_score(y_test, singleLayerPerceptron.predict(X_test), average='macro'))
print('Recall for Single layer Perceptron:',metrics.recall_score(y_test, singleLayerPerceptron.predict(X_test), average='macro'))
print('F1-measure for Single layer Perceptron:',metrics.f1_score(y_test, singleLayerPerceptron.predict(X_test), average='macro'))
# Рассчет метрик качества для Логистической регрессии
print('Accuracy for Logistic Regression:',metrics.accuracy_score(y_test, logisticRegression.predict(X_test)))
print('Precision for Logistic Regression:',metrics.precision_score(y_test, logisticRegression.predict(X_test), average='macro'))
print('Recall for Iris Logistic Regression:',metrics.recall_score(y_test, logisticRegression.predict(X_test), average='macro'))
print('F1-measure for Logistic Regression:',metrics.f1_score(y_test, logisticRegression.predict(X_test), average='macro'))
