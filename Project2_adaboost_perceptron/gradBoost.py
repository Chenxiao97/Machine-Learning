import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

def Error_rate(X_test,Y_test,grid):
    s = X_test.shape[0]
    n = 0
    Y_p = grid.predict(X_test)
    for i in range(s):
        if (Y_p[i] != Y_test[i]):
            n += 1
    score = grid.score(X_test,Y_test)
    return 1-score 

f = pd.read_csv('hw2_data_2.txt',sep='\t', header = None, skiprows = 1)
X_train = f.loc[:699,:19].values
Y_train = f.loc[:699,20].values
X_test = f.loc[700:,:19].values
Y_test = f.loc[700:,20].values

grid = GradientBoostingClassifier(loss = 'deviance')
grid.fit(X_train,Y_train)

importances = grid.feature_importances_
indices = np.argsort(importances)[::-1]

#Print feature ranking
print("Feature ranking:")
for f in range(X_test.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
error_rate = Error_rate(X_test,Y_test,grid)
print('gradient boosting: \nerror rate = {:f} \n'.format(error_rate))


