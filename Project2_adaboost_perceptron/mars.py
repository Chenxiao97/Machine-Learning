import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyearth import Earth

def Error_rate(X_test,Y_test,model):
    s = X_test.shape[0]
    n = 0
    Y_p = model.predict(X_test)
    y_median = median(Y_p)
    for i in range(s):
        if (Y_p[i] <= y_median):
            y_p = -1
        else:
            y_p = 1
        if (y_p != Y_test[i]):
            n += 1
    return n/float(s) 

def median(Y_test):
    '''
    :param Y_test:  (n*1),predict Y, all data labels
    :return: median value of all predict Y
    '''
    y_median = np.median(Y_test)
    return y_median

f = pd.read_csv('hw2_data_2.txt',sep='\t', header = None, skiprows = 1)
X_train = f.loc[:699,:19].values
Y_train = f.loc[:699,20].values
X_test = f.loc[700:,:19].values
Y_test = f.loc[700:,20].values


model = Earth() 
model.fit(X_train,Y_train)

error_rate = Error_rate(X_test,Y_test,model)
print('MARS: \nerror rate = {:f} \n'.format(error_rate))
