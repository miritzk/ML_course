from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# todo: change x below
# todo: download pyclustering package using the command (in the terminal) : pip install pyclustering
x = 'your dataset without the target variable - (ndarray - numpy array in shape (samples, features))'

metric = distance_metric(type_metric.GOWER, max_range=x.max(axis=0))

dbi_list = []
sil_list = []

max_n_clusters = 8

# this part should take you about 40-60 minutes of calculations (maybe more - depends on your computer)
for n_clusters in tqdm(range(2, max_n_clusters, 1)):
    initial_medoids = kmeans_plusplus_initializer(x, n_clusters).initialize(return_index=True)
    kmedoids_instance = kmedoids(x, initial_medoids, metric=metric)
    kmedoids_instance.process()
    assignment = kmedoids_instance.predict(x)

    sil = silhouette_score(x, assignment)
    dbi = davies_bouldin_score(x, assignment)

    dbi_list.append(dbi)
    sil_list.append(sil)


plt.plot(range(2, max_n_clusters, 1), sil_list, marker='o')
plt.title("Silhouette")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(range(2, max_n_clusters, 1), dbi_list, marker='o')
plt.title("Davies-bouldin")
plt.xlabel("Number of clusters")
plt.show()
