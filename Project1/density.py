import pandas as pd
import matplotlib.pyplot as plt

trainingfile = 'HW_1_training.txt'
data = pd.read_csv(trainingfile, delimiter = "\t", header = None, skiprows = 1)

testingfile = 'HW_1_testing.txt'
test = pd.read_csv(testingfile, delimiter = "\t", header = None, skiprows = 1)
import numpy as np
from sklearn.neighbors.kde import KernelDensity

X = test.loc[:,[0,1]].values
Y = test.loc[:,[2]].values 

x1 = data.loc[lambda f : f[2] == 0,[0,1]].values
x2 = data.loc[lambda f : f[2] == 1,[0,1]].values


bandwidth = 0.1
k1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x1)  
den1 = 10**(k1.score_samples(X))

k2 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x2)  
den2 = 10**(k2.score_samples(X))

#predict
s = X.shape[0]
Y_pred = np.zeros((s,1))
for i in range(s):
    sum = den1[i]*0.5+den2[i]*0.5
    k = []
    k.append(den1[i]*0.5/float(sum))
    k.append(den2[i]*0.5/float(sum))
    Y_pred[i] = np.argmax(k)

countWrong = 0
error_rate = 0
for i in range(len(Y)):
	if (Y[i] != Y_pred[i]):
		countWrong += 1

error_rate = countWrong / float(len(Y))

print('Bandwidth={:f}, \nError rate = {:f} \n'.format(bandwidth, error_rate))

wrong0 = []
wrong1 = []
right0 = [] 
right1 = []
for i in range(len(Y)):
    if (Y[i] != Y_pred[i] and Y_pred[i] == 0):        
        wrong0.append(X[i,:])
    elif (Y[i] != Y_pred[i] and Y_pred[i] == 1):      
        wrong1.append(X[i,:])
    elif (Y[i] == Y_pred[i] and Y_pred[i] == 0):      
        right0.append(X[i,:])                      
    else:
        right1.append(X[i,:])

wrong0 = np.matrix(wrong0)
right0 = np.matrix(right0)
wrong1 = np.matrix(wrong1)
right1 = np.matrix(right1)



plt.scatter([right0[:,0]],[right0[:,1]],30, color='blue', marker='o', label='Y = 0')
plt.scatter([right1[:,0]],[right1[:,1]],30, color='blue', marker='X', label='Y = 1')
plt.scatter([wrong0[:,0]],[wrong0[:,1]],30, color='yellow', marker='o', label='Y = 0,misclassified')
plt.scatter([wrong1[:,0]],[wrong1[:,1]],30, color='yellow', marker='X', label='Y = 1,misclassified')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='lower right',
       fontsize = 7)
plt.savefig('Bandwidth={:f}.eps'.format(bandwidth),format='eps')
plt.show()

