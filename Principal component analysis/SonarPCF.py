import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
import pandas as pd
pd.set_option('display.max_columns', 100)


def variance(df):
    # Общее среднее значение
    totalMean = 0
    totalSum = 0
    for i in df.columns:
        mean = 0
        count = 0
        for m in df[i].index:
            mean = mean + (count * df[i][m]) / np.sum(df[i])
            count = count+1
        totalMean = totalMean + mean
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
            mean = mean+(count*df[i][m])/np.sum(df[i])
            count = count+1

        count = 0
        for j in df[i].index:
            totalIngroupp = totalIngroupp + np.power(count - mean, 2) * df[i][j] / np.sum(df[i])
            count = count+1
        print("Внутригрупповая дисперсия группы ", i, ": ", totalIngroupp)

        # Остаточная дисперсия
        totalResVar = totalResVar + (totalIngroupp * np.sum(df[i])) / totalSum

        # Межгрупповая дисперсия
        totalOutgroupp = totalOutgroupp + np.power(mean - totalMean, 2) * np.sum(df[i]) / totalSum

    print("Остаточная дисперсия: ", totalResVar)
    print("Межгрупповая дисперсия: ", totalOutgroupp)
    # Общая дисперсия
    print("Общая дисперсия: ", totalResVar + totalOutgroupp)

# Получение датасета и построение изначальной матрицы корреляции
auditDataSet = pd.read_csv(r"D:\PyCharm\MISIS-NN-ML\Principal component analysis\Datasets\sonar.csv", header=None)

corrMatrix = plt.matshow(auditDataSet.corr())
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Первоначальная матрица корреляции датасета Audit')
plt.show()
# Получение списка матрицы корреляции для выясления элементов со слабейшей корреляцией
df = pd.DataFrame(auditDataSet).dropna()
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
nAuditDataSet = auditDataSet
pcaAuditDataSet = auditDataSet
ilocColumn = result_bottom.iloc[:, 0]
for i in pcaAuditDataSet.columns:
    pcaAuditDataSet = pcaAuditDataSet.drop([i], axis=1)
for i in ilocColumn:
    try:
        # print(pcaAuditDataSet.iloc[:, [pcaAuditDataSet.columns.get_loc(i)]])
        nAuditDataSet = nAuditDataSet.drop([i], axis=1)
        pcaAuditDataSet = pcaAuditDataSet.join(auditDataSet.iloc[:, [auditDataSet.columns.get_loc(i)]])
    except Exception: pass
# print(nAuditDataSet)
# print(pcaAuditDataSet)

# Реализация PCA и присоединение получившихся столбцов к изначальному датасету
pcaAuditDataSet = pd.DataFrame(pcaAuditDataSet).dropna()
pca = decomposition.PCA()
pcaAuditDataSet_transformed = pca.fit(pcaAuditDataSet).transform(pcaAuditDataSet)

mainCompType = 4
pca1 = decomposition.PCA(mainCompType)
pcaAuditDataSet_transformed = pca1.fit(pcaAuditDataSet).transform(pcaAuditDataSet)

principalDf = pd.DataFrame(data = pcaAuditDataSet_transformed).add_prefix('MC_')
nAuditDataSet = nAuditDataSet.join(principalDf)
# print(principalDf)
# print(nAuditDataSet)

#Матрица измененного первоначального датасета
corrAuditDataSet = nAuditDataSet.corr()
# print(corrAuditDataSet)
corrMatrix = plt.matshow(corrAuditDataSet)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Матрица корреляции датасета Audit после использования PCA')
plt.show()

#Матрица факторного анализа
dfCorrAuditMainComp = pd.DataFrame(corrAuditDataSet.iloc[:-1, -mainCompType:]).dropna()
# print(dfCorrAuditMainComp)
variance(dfCorrAuditMainComp)



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