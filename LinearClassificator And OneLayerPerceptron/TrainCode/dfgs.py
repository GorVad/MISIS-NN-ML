
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression


wineTrainDataSet = pd.read_csv("D:\PyCharm\MISIS-NN-ML\LinearClassificator And OneLayerPerceptron\DataSets\wineTrainDataSet.csv", header=None)
wineTestDataSet = pd.read_csv("D:\PyCharm\MISIS-NN-ML\LinearClassificator And OneLayerPerceptron\DataSets\wineTestDataSet.csv", header=None)

yTrain = wineTrainDataSet
XTrain = yTrain.iloc[0:92, 5:7].values
yTrain = yTrain.iloc[0:92, 0].values
yTest = wineTestDataSet
xTest = yTest.iloc[0:86, 5: 7].values
yTest = yTest.iloc[0:86, 0].values

# # standardize training data
XTrain_std = (XTrain - XTrain.mean(axis=0)) / XTrain.std(axis=0)
XTest_std = (xTest - xTest.mean(axis=0)) / xTest.std(axis=0)

singleLayerPerceptron = Perceptron()
singleLayerPerceptron.fit(XTrain, yTrain)

logisticRegression = LogisticRegression()
logisticRegression.fit(XTrain, yTrain)

fig = plot_decision_regions(X=XTrain_std, y=yTrain, clf=singleLayerPerceptron, legend=2)
plt.title('Однойслойный перцептрон. Тренировочная выборка')
plt.show()
fig = plot_decision_regions(X=XTest_std, y=yTest, clf=singleLayerPerceptron, legend=2)
plt.title('Однойслойный перцептрон. Тестовая выборка')
plt.show()

fig = plot_decision_regions(X=XTrain, y=yTrain, clf=logisticRegression, legend=2)
plt.title('Логистическая регрессия. Тренировочная выборка')
plt.show()
fig = plot_decision_regions(X=xTest, y=yTest, clf=logisticRegression, legend=2)
plt.title('Логистическая регрессия. Тестовая выборка')
plt.show()