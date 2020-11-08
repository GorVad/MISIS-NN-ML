import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
import pandas as pd
pd.set_option('display.max_columns', 100)


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

    sottedBarDictionary = {k: v for k, v in sorted(barDictionary.items(), key=lambda item: item[1], reverse=True)}
    keys = sottedBarDictionary.keys()
    values = sottedBarDictionary.values()
    plt.bar(keys, values)

# Получение датасета и построение изначальной матрицы корреляции
SonarDataSet = pd.read_csv(r"D:\PyCharm\MISIS-NN-ML\Principal component analysis\Datasets\sonar.csv", header=None)

corrMatrix = plt.matshow(SonarDataSet.corr())
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Первоначальная матрица корреляции датасета Sonar')
plt.show()
# Получение списка матрицы корреляции для выясления элементов со слабейшей корреляцией
df = pd.DataFrame(SonarDataSet).dropna()
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
nSonarDataSet = SonarDataSet
pcaSonarDataSet = SonarDataSet
ilocColumn = result_bottom.iloc[:, 0]
for i in pcaSonarDataSet.columns:
    pcaSonarDataSet = pcaSonarDataSet.drop([i], axis=1)
for i in ilocColumn:
    try:
        # print(pcaSonarDataSet.iloc[:, [pcaSonarDataSet.columns.get_loc(i)]])
        nSonarDataSet = nSonarDataSet.drop([i], axis=1)
        pcaSonarDataSet = pcaSonarDataSet.join(SonarDataSet.iloc[:, [SonarDataSet.columns.get_loc(i)]])
    except Exception: pass
# print(nSonarDataSet)
print(pcaSonarDataSet)

# Реализация PCA и присоединение получившихся столбцов к изначальному датасету
pcaSonarDataSet = pd.DataFrame(pcaSonarDataSet).dropna()
pca = decomposition.PCA()
pcaSonarDataSet_transformed = pca.fit(pcaSonarDataSet).transform(pcaSonarDataSet)

# # По критерию Кайзера
# mainCompType = --
# По критерию сломанной трости
# mainCompType = 10
# # По критерию каменистой осыпи
mainCompType = 12
pca1 = decomposition.PCA(mainCompType)
pcaSonarDataSet_transformed = pca1.fit(pcaSonarDataSet).transform(pcaSonarDataSet)

principalDf = pd.DataFrame(data = pcaSonarDataSet_transformed).add_prefix('MC_')
nSonarDataSet = nSonarDataSet.join(principalDf)
# print(principalDf)
# print(nSonarDataSet)

#Матрица измененного первоначального датасета
corrSonarDataSet = nSonarDataSet.corr()
# print(corrSonarDataSet)
corrMatrix = plt.matshow(corrSonarDataSet)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Матрица корреляции датасета Sonar после использования PCA')
plt.show()

#Матрица факторного анализа
dfCorrSonarMainComp = pd.DataFrame(corrSonarDataSet.iloc[:-1, -mainCompType:]).dropna()
# print(dfCorrSonarMainComp)
variance(dfCorrSonarMainComp)

# Анализ остатков
reconstruct = pca1.inverse_transform(pcaSonarDataSet_transformed)
residual=pcaSonarDataSet-reconstruct
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