import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def Error_rate(X2,Y2,forest):
    s = X2.shape[0]
    n = 0
    Y_p = forest.predict(X2)
    for i, y in enumerate(Y_p):
        if (y != Y2[i]):
            n += 1
    return n/float(s) 

f = pd.read_csv('hw2_data_2.txt',sep='\t', header = None, skiprows = 1)
X = f.loc[:699,:19].values
Y = f.loc[:699,20].values
X_test = f.loc[700:,:19].values
Y_test = f.loc[700:,20].values
    
min_estimators = 10
max_estimators = 500
step = 10
number_estimators = (max_estimators - min_estimators)/step
print number_estimators

error_rate = []
n_estimators = []

for i in range(min_estimators, max_estimators + step,step):
    clf = RandomForestClassifier(oob_score=True)#warm_start=True
    clf.set_params(n_estimators=i)
    clf.fit(X, Y)

    oob_error = 1 - clf.oob_score_
    n_estimators.append(i)
    error_rate.append(oob_error)
    print i,oob_error

last_oob_error = error_rate[number_estimators]

stabilize_estimator = 0
for i in range(number_estimators):
    diff_rate = abs((error_rate[i] - last_oob_error)/float(last_oob_error))
    print i, diff_rate
    if (diff_rate <=0.002):
        stabilize_estimator = n_estimators[i] 
        print stabilize_estimator
        break
    
print stabilize_estimator
plt.plot(n_estimators, error_rate)
plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend()
plt.show()

forest = RandomForestClassifier(n_estimators = stabilize_estimator)
forest.fit(X,Y)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(X_test.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
error_rate = Error_rate(X_test,Y_test,forest)
print('Random Forest: \nerror rate = {:f} \n'.format(error_rate))
