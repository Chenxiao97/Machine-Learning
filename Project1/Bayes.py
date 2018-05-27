import pandas as pd
import matplotlib.pyplot as plt

trainingfile = 'HW_1_training.txt'
data = pd.read_csv(trainingfile, delimiter = "\t", header = None, skiprows = 1)

testingfile = 'HW_1_testing.txt'
test = pd.read_csv(testingfile, delimiter = "\t", header = None, skiprows = 1)
import numpy as np


mean_1 = np.zeros((1,2))
covariance_1 = np.zeros((2,2))
mean_2 = np.zeros((1,2))
covariance_2 = np.zeros((2,2))

mean_1 = np.mean(data.loc[lambda f : f[2] == 0,[0,1]].values, axis=0)
covariance_1 = np.cov(data.loc[lambda f : f[2] == 0,[0,1]].values.T)
mean_2 = np.mean(data.loc[lambda f : f[2] == 1,[0,1]].values, axis=0)
covariance_2 = np.cov(data.loc[lambda f : f[2] == 1,[0,1]].values.T)

print 'mean vector of class0: ', mean_1
print 'mean vector of class1: ', mean_2
print 'covariance matrix of class0: ', covariance_1
print 'covariance matrix of class1: ', covariance_2

X = test.loc[:,[0,1]].values
Y = test.loc[:,[2]].values 

def decision_function(x,x_m,x_c,prior_probability):
    sig_inv = np.linalg.inv(x_c)         #decision function
    return -.5*np.log( np.linalg.det(x_c))-.5*np.dot(np.dot((x-x_m),sig_inv),(x-x_m).T) + np.log(prior_probability)


def predict(X,m1,c1,m2,c2,frac1,frac2):
    s = X.shape[0]
    new_Y = np.zeros((s,1))
    X_new = np.zeros((s,2))
    j = 0
    l = s-1
    for i in range(s):
        k = []
        k.append(decision_function(X[i,:],m1,c1,frac1))    
        k.append(decision_function(X[i,:],m2,c2,frac2))    
        #classify
        new_Y[i] = np.argmax(k)      
        if np.argmax(k) == 0:
            X_new[j,:] = X[i,:]     
            j += 1
        elif new_Y[i] == 1:         
            X_new[l,:] = X[i,:]
            l -= 1 
    
    return X_new,new_Y#,j,l
def plot(x,m1,c1,m2,c2,frac1,frac2):
    xx1,xx2 = np.meshgrid(np.arange(-5, 7,0.02 ),
                         np.arange(-4, 4,0.02))
    Z = predict(np.array([xx1.ravel(), xx2.ravel()]).T,m1,c1,m2,c2,frac1,frac2)[1]
    
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1, xx2, Z, alpha=0.3)
    
    plt.scatter(x[:200,0], x[:200,1],30,
            color='yellow', marker='o', label='y=0')
    plt.scatter(x[200:,0], x[200:,1],30,
    color='blue', marker='^', label='y=1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='upper left')
    return 0
#equal prior
X_pred,Y_pred = predict(X,mean_1,covariance_1,mean_2,covariance_2,0.5,0.5)
countWrong = 0
for i in range(len(Y)):
	if (Y[i] != Y_pred[i]):
		countWrong += 1
error_rate_e = countWrong / float(len(Y))
print('Equal prior: \nerror rate = {:f} \n'.format(error_rate_e))
plot(data.loc[:,[0,1]].values,mean_1,covariance_1,mean_2,covariance_2,0.5,0.5)
plt.savefig('Equal prior .eps',format='eps')
plt.show()

#priori calculated from data
X_pred_d,Y_pred_d = predict(X,mean_1,covariance_1,mean_2,covariance_2,200/float(325),125/float(325))
countWrong = 0
for i in range(len(Y)):
	if (Y[i] != Y_pred_d[i]):
		countWrong += 1
error_rate_e = countWrong / float(len(Y))
print('Prior calculated from the data: \nerror rate = {:f} \n'.format(error_rate_e))
plot(data.loc[:,[0,1]].values,mean_1,covariance_1,mean_2,covariance_2,200/float(325),125/float(325))
plt.savefig('Prior calculated from the data.eps',format='eps')
plt.show()