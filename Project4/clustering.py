import pandas as pd 
import numpy as np 
import matplotlib.pyplot as pyplot

from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from scipy.spatial import distance

def bic(centroids, labels, clusters, size, X_trans):
	m,n = X_trans.shape
	cl_var = (1.0 / (m - clusters) / n) * sum([sum(distance.cdist(X_trans[np.where(labels == i)], [centroids[0][i]], 'euclidean')**2) for i in range(clusters)])

	const_term = 0.5 * clusters * np.log(m) * (n+1)
	BIC = np.sum([size[i] * np.log(size[i]) - size[i] * np.log(m) -
		((size[i] * n) / 2) * np.log(2*np.pi*cl_var) -
		((size[i] - 1) * n/ 2) for i in range(clusters)]) - const_term
	
	return(BIC)



#load data
df = pd.read_csv('project_data.txt',sep='\t', header = None, skiprows = 1)
X = np.array(df.iloc[:,1:120])
Y = np.array(df.iloc[:,[121]])
X_trans = np.transpose(X)


#clustering method 1: k means clustering
#use BIC to estimate the k in kmeans (although it is obvious from the graph)
k_val = range(1,11)
k_models = []
for k in k_val:
	res = KMeans(n_clusters = k).fit(X_trans)
	k_models.append(res)
#print (k_models)

bic_vals = []
for km in k_models:
	centroids = [km.cluster_centers_]
	labels = km.labels_
	clusters = km.n_clusters
	size = np.bincount(labels)
	bic_vals.append(bic(centroids,labels,clusters,size, X_trans))

print (bic_vals)
#thus select number of clusters = 3
clf_kmeans = KMeans(n_clusters = 3).fit(X_trans)
print (clf_kmeans.labels_)

clf_agg = AgglomerativeClustering(n_clusters = 3).fit(X_trans)
print (clf_agg.labels_)



