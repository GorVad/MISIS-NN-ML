import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
import pandas as pd
pd.set_option('precision', 3)
pd.set_option('display.max_columns', 100)

# Получение датасета и построение изначальной матрицы корреляции
auditDataSet = pd.read_csv(r"D:\PyCharm\MISIS-NN-ML\Principal component analysis\Datasets\audit.csv")
newAuditDataSet = auditDataSet.drop(['Sector_score', 'LOCATION_ID', 'TOTAL', 'numbers','District_Loss', 'History'], axis=1)

corrMatrix = plt.matshow(newAuditDataSet.corr())
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Первоначальная матрица корреляции датасета Audit')
plt.show()

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
pca = decomposition.PCA(n_components=3)
pcaAuditDataSet_centered = pcaAuditDataSet - pcaAuditDataSet.mean(axis=0)
pcaAuditDataSet_transformed = pca.fit_transform(pcaAuditDataSet_centered)
principalDf = pd.DataFrame(data = pcaAuditDataSet_transformed, columns=['1', '2', '3'])
# print(principalDf)
nAuditDataSet = nAuditDataSet.join(principalDf)
# print(nAuditDataSet)

corrMatrix = plt.matshow(nAuditDataSet.corr())
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Матрица корреляции датасета Audit после использования PCA')
plt.show()

df = pd.DataFrame(nAuditDataSet).dropna()
corrM = df.corr()
np.fill_diagonal(corrM.values, np.nan)
order_bottom = np.argsort(corrM.values, axis=1)[:, :3]
result_bottom = pd.DataFrame(
    corrM.columns[order_bottom],
    columns=['Last1','Last2', 'Last3'],
    index=corrM.index
)
for x in result_bottom.columns:
    result_bottom[x+"_Val"] = corrM.lookup(corrM.index, result_bottom[x])
# print(result_bottom)
