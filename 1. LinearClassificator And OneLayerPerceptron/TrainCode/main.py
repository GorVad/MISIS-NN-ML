import matplotlib.pyplot as plt
import matplotlib.pyplot as pltDefault
import matplotlib.pyplot as pltplt
import pandas
from matplotlib import colors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

# def plot_data(lda, X, y, y_pred, fig_index):
#     splot = plt.subplot(2, 2, fig_index)
#
#     tp = (y == y_pred)  # True Positive
#     tp0, tp1 = tp[y == 0], tp[y == 1]
#     X0, X1 = X[y == 0], X[y == 1]
#     X0_tp, X0_fp = X0[tp0], X0[~tp0]
#     X1_tp, X1_fp = X1[tp1], X1[~tp1]
#
#     # class 0: dots
#     plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='red')
#     plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='x',
#                 s=20, color='#990000')  # dark red
#
#     # class 1: dots
#     plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='blue')
#     plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x',
#                 s=20, color='#000099')  # dark blue
#
#     # class 0 and 1 : areas
#     nx, ny = 200, 100
#     x_min, x_max = plt.xlim()
#     y_min, y_max = plt.ylim()
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
#                          np.linspace(y_min, y_max, ny))
#     Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z[:, 1].reshape(xx.shape)
#     plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
#                    norm=colors.Normalize(0., 1.), zorder=0)
#     plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')
#     return splot


dataSet = pandas.read_csv('../DataSets/wine.csv')
# dataSet.drop(['Age'], axis=1, inplace=True)
# print(dataSet)

X = dataSet.iloc[:, 2:3].values
y = dataSet.iloc[:, 1].values
pltDefault.scatter(X, y)
# print(X)
# print(y)
pltDefault.title('Изначальное множество')
# pltDefault.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
lda.fit(X_train, y_train.astype('int'))
# print(X_train)
# print(y_train)
# pltTrain.scatter(X_train, y_train)
# pltTrain.title('Тренировочное множество')
# pltTrain.show()

# y_prediction = logisticRegression.predict(X_test[2].reshape(1, -1))
# print(X_test, y_test)
# print(X_test, y_test)
# pltTest.scatter(X_test, y_prediction)
# pltTest.title('Тестовое множество')
# pltTest.show()

pltplt.title('что')
for x,y in zip(X_test, y_test):
    y_prediction = lda.predict(x.reshape(-1, 1))
    if y.astype('int') == y_prediction:
        pltplt.scatter(x,y,marker='o', color='green')
    else:
        pltplt.scatter(x, y, marker='o', color='red')

pltplt.show()



# for x,y in zip(X_test, y_test):
#     ytest = y.astype('int')
#     y_prediction = logisticRegression.predict(X_test)
#     if ytest == y_prediction.all():
#         plt.scatter(X_test[x], y_test[y], marker='.', color='red')
    # tp0, tp1 = tp[ytest == 12], tp[ytest == 13]
    # X0, X1 = X[ytest == 12], X[ytest == 13]
    # X0_tp = X0[tp0]
    # X1_tp = X1[tp1]


    # # class 0: dots
    # plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='red')
    #
    # # class 1: dots
    # plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='blue')

# for i, (X, y) in zip(X_test,y_test):
#     y_prediction = logisticRegression.predict([X]).reshape(-1, 1)
#     splot = plot_data(logisticRegression, X[0], y, y_prediction, fig_index=2 * i + 1)
#     plt.axis('tight')

# for X,y in zip(X_test,y_test):
#     print (X[0],y)
# plt.tight_layout()
