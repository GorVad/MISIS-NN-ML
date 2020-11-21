from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
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
hcvDataSet = pd.read_csv("D:\Development\PyCharm\MISIS-NN-ML\Clustering\DataSets\hcvdat0.csv").dropna()
XhcvDataSet = hcvDataSet.replace('m', 0).replace('f', 1).drop(columns = ['Category', 'Unnamed: 0'])
pca = decomposition.PCA(2)
pcaXhcvDataSet_transformed = pca.fit(XhcvDataSet).transform(XhcvDataSet)

YhcvDataSet = hcvDataSet.replace('0=Blood Donor', 0).replace('0s=suspect Blood Donor', 1).replace('1=Hepatitis', 2).replace('2=Fibrosis',3).replace('3=Cirrhosis',4)
YhcvDataSet = YhcvDataSet.iloc[:, 1].to_numpy()

optimalClusterClasses(pcaXhcvDataSet_transformed) # Оптимальное количество - 5 кластеров

# k-means
km = KMeans(n_clusters=3)
yKM = km.fit_predict(pcaXhcvDataSet_transformed)
clusterVisualize(yKM, pcaXhcvDataSet_transformed, km)
print(homogeneity_completeness_v_measure(YhcvDataSet, yKM))
print(mutual_info_score(YhcvDataSet, yKM))

# KMedoids - Неиерархический, итеративный метод
kMedoids = KMedoids(n_clusters=3, metric = 'euclidean')
yminiKM = kMedoids.fit_predict(X = pcaXhcvDataSet_transformed)
clusterVisualize(yminiKM, pcaXhcvDataSet_transformed, kMedoids)
print(homogeneity_completeness_v_measure(YhcvDataSet, yminiKM))
print(mutual_info_score(YhcvDataSet, yminiKM))

kMedoids = KMedoids(n_clusters=3, metric = 'manhattan')
yminiKM = kMedoids.fit_predict(X = pcaXhcvDataSet_transformed)
clusterVisualize(yminiKM, pcaXhcvDataSet_transformed, kMedoids)
print(homogeneity_completeness_v_measure(YhcvDataSet, yminiKM))
print(mutual_info_score(YhcvDataSet, yminiKM))

# AgglomerativeClustering - Иерархический агломеративный метод
acSingleEUC = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
yACSingleEUC = acSingleEUC.fit_predict(X = pcaXhcvDataSet_transformed)
clusterVisualize(yACSingleEUC, pcaXhcvDataSet_transformed, acSingleEUC)
print(homogeneity_completeness_v_measure(YhcvDataSet, yACSingleEUC))
print(mutual_info_score(YhcvDataSet, yACSingleEUC))

acSingleMAN = AgglomerativeClustering(n_clusters=3, affinity='manhattan', linkage='complete')
yACSingleMAN = acSingleMAN.fit_predict(X = pcaXhcvDataSet_transformed)
clusterVisualize(yACSingleMAN, pcaXhcvDataSet_transformed, acSingleMAN)
print(homogeneity_completeness_v_measure(YhcvDataSet, yACSingleEUC))
print(mutual_info_score(YhcvDataSet, yACSingleEUC))
