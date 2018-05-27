import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
def plotclassifier(grid,para_range,label):
    '''
    :param grid: grid
    :param C_range : range of C parameter
    :param para_range : range of tuning parameter
    :param label : string, name of parameter
    :return: plot of score vs parameter
    '''
    scores = [x[1] for x in grid.grid_scores_]

    #plot
    plt.plot(para_range, scores)

    plt.legend()
    plt.xlabel(label)
    plt.ylabel('Mean score')
    plt.show()



 

#load data
df = pd.read_csv('project_data.txt',sep='\t', header = None, skiprows = 1)
X = np.array(df.iloc[:,1:120])
Y = np.array(df.iloc[:,[121]])
X_trans = np.transpose(X)

#Split the data in a 3:1:1 ratio into training, validation, and testing sets.
#60:20:20
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


# #LogisticRegressionCV
# clf_log = LogisticRegressionCV(cv = 10)
# clf_log.fit(X_train,y_train)
# score_log = clf_log.score(X_val,y_val)
# print('LogisticRegressionCV Validation = {:f} \n'.format(score_log))



# estimators = np.arange(10,1000,100)
# parameters = dict(n_estimators = estimators)
# grid_forest = GridSearchCV(RandomForestClassifier(oob_score = True),param_grid = parameters, cv = 10)
# col, row = y_train.shape
# y_train_2 = y_train.reshape(col,)
# grid_forest.fit(X_train, y_train_2)
# best_forset = grid_forest.best_estimator_
# print("BEST by gridcv: ", best_forset)
# best_forset.fit(X_train,y_train)
# score_forest = best_forset.score(X_val,y_val)
# print('Random Forest Validation: {:f} \n'.format(score_forest))
# print('parameter tuning:')
# for x in grid_forest.grid_scores_:
# 	print (x[1])

# imp_forest = best_forset.feature_importances_
# indices = np.argsort(imp_forest)[::-1]
# print("Imoprtance:")
# for f in range(X_train.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], imp_forest[indices[f]]))


       

#GradientBoostingClassifier
lr_s = np.arange(1e-2,1,1e-2)
parameters = dict(learning_rate=lr_s)
grid_gb = GridSearchCV(GradientBoostingClassifier(),param_grid = parameters, cv = 10)
col, row = y_train.shape
y_train_2 = y_train.reshape(col,)
grid_gb.fit(X_train, y_train_2)
best_gb = grid_gb.best_estimator_
print("BEST by gridcv: ", best_gb)
best_gb.fit(X_train,y_train)
score_gb = best_gb.score(X_val,y_val)
print('Gradient Boosting validation: {:f} \n'.format(score_gb))
print('parameter tuning:')
for x in grid_gb.grid_scores_:
	print (x[1])

imp_gb = best_gb.feature_importances_
indices = np.argsort(imp_gb)[::-1]
print("Importance:")
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], imp_gb[indices[f]]))
plotclassifier(grid_gb,lr_s,'learning_rate')


final_score = best_gb.score(X_test,y_test)
print ('Accuracy on Test data by gradient boosting {:f} \n'.format(final_score))

