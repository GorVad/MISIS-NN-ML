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
order_bottom = np.argsort(corrM.values, axis=1)[:, :3]
result_bottom = pd.DataFrame(
    corrM.columns[order_bottom],
    columns=['Last1','Last2', 'Last3'],
    index=corrM.index
)
for x in result_bottom.columns:
    result_bottom[x+"_Val"] = corrM.lookup(corrM.index, result_bottom[x])
# print(result_bottom)

# Выдергивание атрибутов со слабой корреляции из общего датасета
pcaAuditDataSet = auditDataSet.iloc[:, [6, 2, 12, 14, 16, 17, 20, 23]]
nAuditDataSet = newAuditDataSet.drop(['PROB','RiSk_E', 'Risk_F', 'PARA_A', 'CONTROL_RISK', 'Money_Value', 'Score_B', 'Risk_D'], axis=1)
# print(nAuditDataSet)

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
print(result_bottom)
