import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
import pandas as pd
pd.set_option('precision', 3)
pd.set_option('display.max_columns', 100)

# Получение датасета и построение изначальной матрицы корреляции
auditDataSet = pd.read_csv(r"D:\PyCharm\MISIS-NN-ML\Principal component analysis\Datasets\audit.csv")
newAuditDataSet = auditDataSet.drop(['Sector_score', 'LOCATION_ID', 'TOTAL', 'numbers','District_Loss', 'History'], axis=1)

# corrMatrix = plt.matshow(newAuditDataSet.corr())
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Первоначальная матрица корреляции датасета Audit')
# plt.show()

# Получение списка матрицы корреляции для выясления элементов со слабейшей корреляцией
df = pd.DataFrame(newAuditDataSet).dropna()
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
nAuditDataSet = newAuditDataSet
pcaAuditDataSet = newAuditDataSet
ilocColumn = result_bottom.iloc[:, 0]
for i in pcaAuditDataSet.columns:
    pcaAuditDataSet = pcaAuditDataSet.drop([i], axis=1)
for i in ilocColumn:
    try:
        # print(pcaAuditDataSet.iloc[:, [pcaAuditDataSet.columns.get_loc(i)]])
        nAuditDataSet = nAuditDataSet.drop([i], axis=1)
        pcaAuditDataSet = pcaAuditDataSet.join(newAuditDataSet.iloc[:, [newAuditDataSet.columns.get_loc(i)]])
    except Exception: pass
# print(nAuditDataSet)
# print(pcaAuditDataSet)

# Реализация PCA и присоединение получившихся столбцов к изначальному датасету
pcaAuditDataSet = pd.DataFrame(pcaAuditDataSet).dropna()
pca = decomposition.PCA()
# pcaAuditDataSet_centered = pcaAuditDataSet - pcaAuditDataSet.mean(axis=0)
pcaAuditDataSet_transformed = pca.fit(pcaAuditDataSet).transform(pcaAuditDataSet)

# principalDf = pd.DataFrame(data = pcaAuditDataSet_transformed, columns=['1', '2'])
# print(principalDf)
# nAuditDataSet = nAuditDataSet.join(principalDf)
# print(nAuditDataSet)

# corrMatrix = plt.matshow(nAuditDataSet.corr())
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Матрица корреляции датасета Audit после использования PCA')
# plt.show()

df = pd.DataFrame(nAuditDataSet).dropna()
corrM = df.corr()
np.fill_diagonal(corrM.values, np.nan)
order_bottom = np.argsort(corrM.values, axis=1)[:, :1]
result_bottom = pd.DataFrame(
    corrM.columns[order_bottom],
    columns=['Last1'],
    index=corrM.index
)
for x in result_bottom.columns:
    result_bottom[x+"_Val"] = corrM.lookup(corrM.index, result_bottom[x])
# print(result_bottom)

print(pca.explained_variance_ratio_)

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