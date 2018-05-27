import numpy as np
import pandas as pd

def predict(x,w,y):
    y_p = np.dot(x,w)*y
    return y_p

def perceptron(X,w,Y,lrate):
    flag = True
    while(flag):
        total_error = 0       
        for i, x in enumerate(X):
            x = np.insert(x,2,1)  #Add bias = 1 in x
            y_p = predict(x,w,Y[i])
            if (y_p <= 0):
                total_error += 1
                w = w + lrate*x*Y[i]
        if (total_error == 0):
            flag = False  #loop until all training data are correct
    return w

def Error_rate(X2,w,Y2):
    s = Y2.shape[0]
    n = 0
    for i, x in enumerate(X2):
        x2 = np.insert(x,2,1)
        y2_p = predict(x2,w,Y2[i])
        if (y2_p <= 0):
            n += 1
    return n/float(s)   
    

f = pd.read_csv('hw2_data_1.txt',sep='\t', header = None, skiprows = 1)
X1 = f.loc[:69,[0,1]].values
Y1 = f.loc[:69,[2]].values
X2 = f.loc[70:,[0,1]].values
Y2 = f.loc[70:,[2]].values

lrate = 1
w = np.ones(3)
w = perceptron(X,w,Y,lrate)

error_rate = Error_rate(X2,w,Y2)
print('perceptron: \nerror rate = {:f} \n'.format(error_rate))
