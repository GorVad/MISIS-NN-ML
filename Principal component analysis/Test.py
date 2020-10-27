import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v

auditDataSet = pd.read_csv(r"D:\PyCharm\MISIS-NN-ML\Principal component analysis\Datasets\audit.csv")
corrMatrix = plt.matshow(auditDataSet.corr())
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.show()

df = pd.DataFrame(auditDataSet).dropna()
corrM = df.corr()
np.fill_diagonal(corrM.values, np.nan)
print(corrM)

order_bottom = np.argsort(corrM.values, axis=1)[:, :3]
result_bottom = pd.DataFrame(
    corrM.columns[order_bottom],
    columns=['Last1','Last2', 'Last3'],
    index=corrM.index
)
for x in result_bottom.columns:
    result_bottom[x+"_Val"] = corrM.lookup(corrM.index, result_bottom[x])
print(result_bottom)
# print(result_bottom)
# auditDataSet = pandas.read_csv(r"D:\PyCharm\MISIS-NN-ML\Principal component analysis\Datasets\isolet.csv")
# plt.matshow(auditDataSet.corr())
# plt.show()
# auditDataSet = pandas.read_csv(r"D:\PyCharm\MISIS-NN-ML\Principal component analysis\Datasets\sonar.csv")
# plt.matshow(auditDataSet.corr())
# plt.show()