import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
import pandas as pd
pd.set_option('display.max_columns', 10)


def variance(df):
    barDictionary = {}
    # Общее среднее значение
    totalMean = 0
    totalSum = 0
    totalCount = 0
    for i in df.columns:
        mean = 0
        count = 0
        for m in df[i].index:
            mean = mean + (count * df[i][m]) / np.sum(df[i])
            count = count + 1
        totalMean = totalMean + mean
        totalCount = totalCount + count
        totalSum = totalSum = np.sum(df[i])

    totalResVar = 0
    totalOutgroupp = 0
    totalIngroupp = 0
    for i in df.columns:

        # Внутригрупповая дисперсия
        totalIngroupp = 0
        mean = 0
        count = 0
        for m in df[i].index:
            mean = mean + (count * df[i][m]) / np.sum(df[i])
            count = count + 1

        count = 0
        for j in df[i].index:
            totalIngroupp = totalIngroupp + np.power(count - mean, 2)
            count = count + 1
        totalIngroupp = totalIngroupp / count
        print("Внутригрупповая дисперсия группы ", i, ": ", totalIngroupp)

        # Остаточная дисперсия
        totalResVar = totalResVar + (totalIngroupp * np.sum(df[i])) / totalSum

        # Межгрупповая дисперсия
        totalOutgroupp = totalOutgroupp + np.power(mean - totalMean, 2) * count / totalCount
        barDictionary[i] = totalIngroupp

    print("Остаточная дисперсия: ", totalResVar)
    print("Межгрупповая дисперсия: ", totalOutgroupp)
    # Общая дисперсия
    print("Общая дисперсия: ", totalResVar + totalOutgroupp)

    sottedBarDictionary ={k: v for k, v in sorted(barDictionary.items(), key=lambda item: item[1], reverse=True)}
    keys = sottedBarDictionary.keys()
    values = sottedBarDictionary.values()
    plt.bar(keys, values)

# Получение датасета и построение изначальной матрицы корреляции
IsoletDataSet = pd.read_csv(r"D:\PyCharm\MISIS-NN-ML\Principal component analysis\Datasets\isolet.csv", header=None)

corrMatrix = plt.matshow(IsoletDataSet.corr())
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Первоначальная матрица корреляции датасета Isolet')
plt.show()
# Получение списка матрицы корреляции для выясления элементов со слабейшей корреляцией
df = pd.DataFrame(IsoletDataSet).dropna()
corrM = df.corr()
np.fill_diagonal(corrM.values, np.nan)
order_bottom = np.argsort(corrM.values, axis=1)[:, :1]
result_bottom = pd.DataFrame(
    corrM.columns[order_bottom],
    columns=['Lowest cor'],
    index=corrM.index
)
for x in result_bottom.columns:
    result_bottom[x+"_Val"] = corrM.lookup(corrM.index, result_bottom[x])
# print(result_bottom)

# Создание DataFrame для PCA - 1. Отдельный из атрибутов со слабой корреляцией и 2. Изначальный без слабых
nIsoletDataSet = IsoletDataSet
pcaIsoletDataSet = IsoletDataSet
ilocColumn = result_bottom.iloc[:, 0]
for i in pcaIsoletDataSet.columns:
    pcaIsoletDataSet = pcaIsoletDataSet.drop([i], axis=1)
for i in ilocColumn:
    try:
        # print(pcaIsoletDataSet.iloc[:, [pcaIsoletDataSet.columns.get_loc(i)]])
        nIsoletDataSet = nIsoletDataSet.drop([i], axis=1)
        pcaIsoletDataSet = pcaIsoletDataSet.join(IsoletDataSet.iloc[:, [IsoletDataSet.columns.get_loc(i)]])
    except Exception: pass
# print(nIsoletDataSet)
# print(pcaIsoletDataSet)

# Реализация PCA и присоединение получившихся столбцов к изначальному датасету
pcaIsoletDataSet = pd.DataFrame(pcaIsoletDataSet).dropna()
pca = decomposition.PCA()
pcaIsoletDataSet_transformed = pca.fit(pcaIsoletDataSet).transform(pcaIsoletDataSet)

# # По критерию Кайзера
mainCompType = 21
# По критерию сломанной трости
# mainCompType = 15
# # По критерию каменистой осыпи
# mainCompType = 12

pca1 = decomposition.PCA(mainCompType)
pcaIsoletDataSet_transformed = pca1.fit(pcaIsoletDataSet).transform(pcaIsoletDataSet)

principalDf = pd.DataFrame(data = pcaIsoletDataSet_transformed).add_prefix('MC_')
nIsoletDataSet = nIsoletDataSet.join(principalDf)
# print(principalDf)
# print(nIsoletDataSet)

#Матрица измененного первоначального датасета
corrIsoletDataSet = nIsoletDataSet.corr()
# print(corrIsoletDataSet)
corrMatrix = plt.matshow(corrIsoletDataSet)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Матрица корреляции датасета Isolet после использования PCA')
plt.show()

#Матрица факторного анализа
dfCorrIsoletMainComp = pd.DataFrame(corrIsoletDataSet.iloc[:-1, -mainCompType:]).dropna()
print(dfCorrIsoletMainComp)
variance(dfCorrIsoletMainComp)

# Анализ остатков
reconstruct = pca1.inverse_transform(pcaIsoletDataSet_transformed)
residual=pcaIsoletDataSet-reconstruct
print(residual)
print("ERV:", sum(pca1.explained_variance_ratio_))

def scree_plot():
    ax = figure().gca()
    ax.plot(pca.explained_variance_)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.axhline(y=1, linewidth=1, color='r', alpha=0.5)
    plt.title('Scree Plot of PCA: Component Eigenvalues')
    show()
scree_plot()

def var_explained():
    ax = figure().gca()
    ax.plot(np.cumsum(pca.explained_variance_ratio_))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.axvline(x=1, linewidth=1, color='r', alpha=0.5)
    plt.title('Explained Variance of PCA by Component')
    show()
var_explained()