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
CD4DataSet = make_blobs(n_samples=10000, cluster_std=3.5, centers=3)
XCD4DataSet, YCD4DataSet = CD4DataSet
pca = decomposition.PCA(2)
pcaXCD4DataSet_transformed = pca.fit(XCD4DataSet).transform(XCD4DataSet)

optimalClusterClasses(pcaXCD4DataSet_transformed) # Оптимальное количество - 5 кластеров

# k-means
km = KMeans(n_clusters=3)
yKM = km.fit_predict(pcaXCD4DataSet_transformed)
clusterVisualize(yKM, pcaXCD4DataSet_transformed, km)
print(homogeneity_completeness_v_measure(YCD4DataSet, yKM))
print(homogeneity_score(YCD4DataSet, yKM))
print(completeness_score(YCD4DataSet, yKM))
print(v_measure_score(YCD4DataSet, yKM))

# AgglomerativeClustering - Неиерархический, итеративный метод
miniKM = MiniBatchKMeans(n_clusters=3)
yminiKM = miniKM.fit_predict(pcaXCD4DataSet_transformed)
clusterVisualize(yminiKM, pcaXCD4DataSet_transformed, miniKM)
print(homogeneity_completeness_v_measure(YCD4DataSet, yminiKM))
print(homogeneity_score(YCD4DataSet, yminiKM))
print(completeness_score(YCD4DataSet, yminiKM))
print(v_measure_score(YCD4DataSet, yminiKM))

# AgglomerativeClustering - Иерархический агломеративный метод
acSingleEUC = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
yACSingleEUC = acSingleEUC.fit_predict(X = pcaXCD4DataSet_transformed)
clusterVisualize(yACSingleEUC, pcaXCD4DataSet_transformed, acSingleEUC)
print(homogeneity_completeness_v_measure(YCD4DataSet, yACSingleEUC))
print(homogeneity_score(YCD4DataSet, yACSingleEUC))
print(completeness_score(YCD4DataSet, yACSingleEUC))
print(v_measure_score(YCD4DataSet, yACSingleEUC))

acSingleMAN = AgglomerativeClustering(n_clusters=3, affinity='manhattan', linkage='complete')
yACSingleMAN = acSingleMAN.fit_predict(X = pcaXCD4DataSet_transformed)
clusterVisualize(yACSingleMAN, pcaXCD4DataSet_transformed, acSingleMAN)
print(homogeneity_completeness_v_measure(YCD4DataSet, yACSingleEUC))
print(homogeneity_score(YCD4DataSet, yACSingleEUC))
print(completeness_score(YCD4DataSet, yACSingleEUC))
print(v_measure_score(YCD4DataSet, yACSingleEUC))
