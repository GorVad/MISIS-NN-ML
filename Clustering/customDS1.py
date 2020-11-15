from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import homogeneity_completeness_v_measure, homogeneity_score, completeness_score, v_measure_score
from sklearn import decomposition
import pandas as pd

pd.set_option('display.max_columns', 10)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def optimalClusterClasses (X):
    mms = MinMaxScaler()
    mms.fit(X)
    data_transformed = mms.transform(X)

    Sum_of_squared_distances = []
    K = range(1, 15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data_transformed)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def clusterVisualize (y_km, X, cMethod):
    plt.scatter(X[:, 0], X[:, 1], c=y_km, s=50, cmap='viridis')
    # centers = cMethod.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

# Подготовка данных
hcvDataSet = make_blobs(n_samples=10000, cluster_std=0.3, centers=5)
XhcvDataSet, YhcvDataSet = hcvDataSet
pca = decomposition.PCA(2)
pcaXhcvDataSet_transformed = pca.fit(XhcvDataSet).transform(XhcvDataSet)

optimalClusterClasses(pcaXhcvDataSet_transformed) # Оптимальное количество - 5 кластеров

# k-means
km = KMeans(n_clusters=3)
yKM = km.fit_predict(pcaXhcvDataSet_transformed)
clusterVisualize(yKM, pcaXhcvDataSet_transformed, km)
print(homogeneity_completeness_v_measure(YhcvDataSet, yKM))
print(homogeneity_score(YhcvDataSet, yKM))
print(completeness_score(YhcvDataSet, yKM))
print(v_measure_score(YhcvDataSet, yKM))

# AgglomerativeClustering - Неиерархический, итеративный метод
miniKM = MiniBatchKMeans(n_clusters=5)
yminiKM = miniKM.fit_predict(pcaXhcvDataSet_transformed)
clusterVisualize(yminiKM, pcaXhcvDataSet_transformed, miniKM)
print(homogeneity_completeness_v_measure(YhcvDataSet, yminiKM))
print(homogeneity_score(YhcvDataSet, yminiKM))
print(completeness_score(YhcvDataSet, yminiKM))
print(v_measure_score(YhcvDataSet, yminiKM))

# AgglomerativeClustering - Иерархический агломеративный метод
acSingleEUC = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
yACSingleEUC = acSingleEUC.fit_predict(X = pcaXhcvDataSet_transformed)
clusterVisualize(yACSingleEUC, pcaXhcvDataSet_transformed, acSingleEUC)
print(homogeneity_completeness_v_measure(YhcvDataSet, yACSingleEUC))
print(homogeneity_score(YhcvDataSet, yACSingleEUC))
print(completeness_score(YhcvDataSet, yACSingleEUC))
print(v_measure_score(YhcvDataSet, yACSingleEUC))

acSingleMAN = AgglomerativeClustering(n_clusters=5, affinity='manhattan', linkage='complete')
yACSingleMAN = acSingleMAN.fit_predict(X = pcaXhcvDataSet_transformed)
clusterVisualize(yACSingleMAN, pcaXhcvDataSet_transformed, acSingleMAN)
print(homogeneity_completeness_v_measure(YhcvDataSet, yACSingleEUC))
print(homogeneity_score(YhcvDataSet, yACSingleEUC))
print(completeness_score(YhcvDataSet, yACSingleEUC))
print(v_measure_score(YhcvDataSet, yACSingleEUC))
