from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import homogeneity_completeness_v_measure, mutual_info_score
from sklearn_extra.cluster import KMedoids
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
CD2DataSet = make_blobs(n_samples=1000, cluster_std=1.1, centers=3)
XCD2DataSet, YCD2DataSet = CD2DataSet
pca = decomposition.PCA(2)
pcaXCD2DataSet_transformed = pca.fit(XCD2DataSet).transform(XCD2DataSet)

optimalClusterClasses(pcaXCD2DataSet_transformed) # Оптимальное количество - 5 кластеров

# k-means
km = KMeans(n_clusters=3)
yKM = km.fit_predict(pcaXCD2DataSet_transformed)
clusterVisualize(yKM, pcaXCD2DataSet_transformed, km)
print(homogeneity_completeness_v_measure(YCD2DataSet, yKM))
print(mutual_info_score(YCD2DataSet, yKM))

# KMedoids - Неиерархический, итеративный метод
kMedoids = KMedoids(n_clusters=3, metric = 'euclidean')
yminiKM = kMedoids.fit_predict(X = pcaXCD2DataSet_transformed)
clusterVisualize(yminiKM, pcaXCD2DataSet_transformed, kMedoids)
print(homogeneity_completeness_v_measure(YCD2DataSet, yminiKM))
print(mutual_info_score(YCD2DataSet, yminiKM))

kMedoids = KMedoids(n_clusters=3, metric = 'manhattan')
yminiKM = kMedoids.fit_predict(X = pcaXCD2DataSet_transformed)
clusterVisualize(yminiKM, pcaXCD2DataSet_transformed, kMedoids)
print(homogeneity_completeness_v_measure(YCD2DataSet, yminiKM))
print(mutual_info_score(YCD2DataSet, yminiKM))

# AgglomerativeClustering - Иерархический агломеративный метод
acSingleEUC = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
yACSingleEUC = acSingleEUC.fit_predict(X = pcaXCD2DataSet_transformed)
clusterVisualize(yACSingleEUC, pcaXCD2DataSet_transformed, acSingleEUC)
print(homogeneity_completeness_v_measure(YCD2DataSet, yACSingleEUC))
print(mutual_info_score(YCD2DataSet, yACSingleEUC))

acSingleMAN = AgglomerativeClustering(n_clusters=3, affinity='manhattan', linkage='complete')
yACSingleMAN = acSingleMAN.fit_predict(X = pcaXCD2DataSet_transformed)
clusterVisualize(yACSingleMAN, pcaXCD2DataSet_transformed, acSingleMAN)
print(homogeneity_completeness_v_measure(YCD2DataSet, yACSingleEUC))
print(mutual_info_score(YCD2DataSet, yACSingleEUC))
