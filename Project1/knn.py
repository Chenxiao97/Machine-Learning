#author: Chenxiao Wang
#date: Feb 21, 2018
#measure similarity
from __future__ import division

import math

def euclideanDistance(instance1, instance2, length):  #manhattan distance or euclidean distance?
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]),2)
	return math.sqrt(distance)




#get neighbours
import operator
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance) - 1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance,trainingSet[x],length)
		distances.append((trainingSet[x],dist))
	distances.sort(key = operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors



#prediction
'''parameters:
	 neighbors: array-like'''
def predict(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sortedVotes[0][0]

#measure accuracy
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0




import pandas as pd
import numpy as np

#from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
#main function

#getting data
import pandas as pd
trainfile = 'HW_1_training.txt'
traindata = pd.read_csv(trainfile, delimiter = "\t")
testfile = 'HW_1_testing.txt'
testdata = pd.read_csv(testfile, delimiter = "\t")

trainingSet =  traindata.as_matrix()
testSet = testdata.as_matrix()
k = 1
predictions = []
print testSet.shape
for x in range(len(testSet)):
	neighbors = getNeighbors(trainingSet, testSet[x], k)
	result = predict(neighbors)
	predictions.append(result)

acc = getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(acc) + '%')
y_actul = []
for x in range(len(testSet)):
	y_actul.append(testSet[x][-1])

y_pred = []
y_pred = predictions
tn, fp, fn, tp = confusion_matrix(y_actul, y_pred).ravel()



print (tn, fp, fn, tp)

sensitivity = float(tp / (tp + fn))
specificity = float(tn / (tn + fp))
falseDiscoveryRate = float(fp / (fp + tp))
print 'sensitivity: ', sensitivity
print 'specificity: ', specificity
print 'falseDiscoveryRate: ', falseDiscoveryRate

	
def misclassified_point(X,Y,y_pred):
    mis_0 = []
    mis_1 = []
    right_0 = []
    right_1 = []
    s = Y.shape[0]
    for i in range(s):
        if (Y[i] != y_pred[i] and y_pred[i] == 0):   
            mis_0.append(X[i,:])
        elif (Y[i] != y_pred[i] and y_pred[i] == 1):
        	mis_1.append(X[i,:])
        elif (Y[i] == y_pred[i] and y_pred[i] == 0): 
            right_0.append(X[i,:])                 
        else:
            right_1.append(X[i,:])
    mis_0 = np.matrix(mis_0)
    right_0 = np.matrix(right_0)
    mis_1 = np.matrix(mis_1)
    right_1 = np.matrix(right_1)
    return mis_0,mis_1,right_0,right_1

f1 = pd.read_csv('HW_1_training.txt',sep='\t', header = None, skiprows = 1)
f2 = pd.read_csv('HW_1_testing.txt',sep='\t', header = None, skiprows = 1)

#read testing data
X = f2.loc[:,[0,1]].values
Y = f2.loc[:,[2]].values

#read traing data
x = f1.loc[:,[0,1]].values
y = f1.loc[:,[2]].values
#8,4,156,92
mis_0,mis_1,right_0,right_1 = misclassified_point(X,Y,y_pred)

plt.scatter(right_0[:,0],right_0[:,1],30,
        color='black', marker='o', label='healthy controls')
plt.scatter(right_1[:,0],right_1[:,1],30,
        color='black', marker='x', label='disease cases')
plt.scatter(mis_0[:,0],mis_0[:,1],30,
        color='red', marker='o', label='healthy controls,mis')
plt.scatter(mis_1[:,0],mis_1[:,1],30,
        color='red', marker='x', label='disease cases,mis')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(scatterpoints=1,
       loc='lower right',
       ncol=1,
       fontsize=7)
plt.savefig('k={:d}.eps'.format(K),format='eps')
plt.show()













