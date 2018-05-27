import pandas as pd 
import numpy as np 
import matplotlib.pyplot as pyplot

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#load data
df = pd.read_csv('project_data.txt',sep='\t', header = None, skiprows = 1)
X = np.array(df.iloc[:,1:120])
Y = np.array(df.iloc[:,[121]])

#subgroups of proteins
X_trans = np.transpose(X)
pca = PCA(n_components = 2)
x_protein = pca.fit_transform(StandardScaler().fit_transform(X_trans))
pyplot.scatter(x_protein[:,0], x_protein[:,1],10, color='red', marker='o')
pyplot.xlabel('x1')
pyplot.ylabel('x2')
pyplot.show()


#subgroups of samples
x_sample = pca.fit_transform(StandardScaler().fit_transform(X))
target_0 = []
target_1 = []
for i,y in enumerate(Y):
	if(y == 0):
		target_0.append(x_sample[i,:])
	else:
		target_1.append(x_sample[i,:])
target_0 = np.array(target_0)
target_1 = np.array(target_1)

pyplot.scatter(target_0[:,0], target_0[:,1], 15, color='magenta', marker='o', label='y = 0')
pyplot.scatter(target_1[:,0], target_1[:,1], 15, color='blue', marker='o', label='y = 1')
pyplot.xlabel('x1')
pyplot.ylabel('x2')
pyplot.legend(loc='upper right')
pyplot.show()


