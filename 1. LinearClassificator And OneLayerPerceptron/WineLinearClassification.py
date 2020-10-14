import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

y = pd.read_csv("1. LinearClassificator And OneLayerPerceptron/DataSets/wine.csv", header=None)
x = y.iloc[0:178, [7, 12]].values
y = y.iloc[0:178, 0].values
y = np.where(y==3, -1, 1)


logreg_clf = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=27)

logreg_clf.fit(X_train,y_train)

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

# Showing the final results of the perceptron model.Test Data
plot_decision_regions(X_test, y_test, classifier=logreg_clf)
plt.title('Тренировочная выборка')
plt.show()
# Showing the final results of the perceptron model.Testing Data
plot_decision_regions(X_train, y_train, classifier=logreg_clf)
plt.title('Тренировочная выборка')
plt.show()