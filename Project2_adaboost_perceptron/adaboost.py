import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def lineclassify(value,x,label):
    if (label == 0):
        if (x <= value):
            y_p = -1      
        else:
            y_p = 1
    elif (label == 1):
        if (x <= value):
            y_p = 1      
        else:
            y_p = -1
           
    return y_p


#Classify all training data with one weak classfier
def Weighterror(value,X,Y,W,label):
    weighterror = 0
    for i, x in enumerate(X):
        y_p = lineclassify(value,x,label)   
        y_dot = y_p*Y[i]
        
        if (y_dot <= 0):      
            weighterror += W[i]
           
    return weighterror

def Sumweight(W):
    sumweight = 0
    for i, w in enumerate(W):
        sumweight += W[i]
    return sumweight

def changeweight(value,X_line,Y,W,label,a):

    for i,x in enumerate(X_line):
        y_p = lineclassify(value,X_line[i],label)
        y_dot = y_p*Y[i]
        
        if (y_dot <= 0): 
            W[i] = W[i]*np.exp(a)#calculate new wi
            
        else:
            continue
    return W

#Find the best xij to classify traning data
def Buildline(X,Y,W):

    weight_x1_l = []
    weight_x1_r = []
    weight_x2_l = []
    weight_x2_r = []
    weight_x = []
    x_best = []
    X_list = []
    
    X1 = X[:,0] #all x1
    X2 = X[:,1] #all x2
    
    X_list.append(X1)
    X_list.append(X1)
    X_list.append(X2)
    X_list.append(X2)

        
    for i, x1 in enumerate(X1):
        weighterror1_l = Weighterror(X1[i],X1,Y,W,0) #weighterror of X1,left
        weight_x1_l.append(weighterror1_l)
        
        weighterror1_r = Weighterror(X1[i],X1,Y,W,1)
        weight_x1_r.append(weighterror1_r) #weighterror of X1,right
    
    weight_x.append(weight_x1_l) #put (l,x1) in list, index = 0
    weight_x.append(weight_x1_r) #put (r,x1) in list, index = 1
    
    
    for i, x2 in enumerate(X2):
        weighterror2_l = Weighterror(X2[i],X2,Y,W,0) #weighterror of X2,left
        weight_x2_l.append(weighterror2_l)
        weighterror2_r = Weighterror(X2[i],X2,Y,W,1) #weighterror of X2,right
        weight_x2_r.append(weighterror2_r)
    
    
    weight_x.append(weight_x2_l) 
    weight_x.append(weight_x2_r)
    weight_x_min_list = np.min(weight_x, axis=1)
    weight_x_min = min(weight_x_min_list) #minimum in all list, smallest weighterror

    index_list = np.argmin(weight_x_min_list)
    label_list = [[0,0],[1,0],[0,1],[1,1]] 
    label = label_list[index_list]
    
    index = weight_x[index_list].index(weight_x_min)

    best_value = X_list[index_list][index]
    
    x_best.append(best_value)
    x_best.append(label)
    return x_best

#Build Adaboost training
def Adaboost(iteration,X,Y,W,X_test,Y_test):

    error_rate = []
   
    Y_testp = np.zeros(Y_test.shape[0])
    for i in range(iteration):
        n = 0
        x_best = Buildline(X,Y,W)

        value, label_index ,line_index = x_best[0],x_best[1][0],x_best[1][1]  
        X_line = X[:,line_index] 
        
        weighterror = Weighterror(value,X_line,Y,W,label_index)
        
        sumweight = Sumweight(W)
        error = weighterror/float(sumweight)  #calculate error
        print ("errorm:")
        print error
        
        a = np.log((1-error)/float(error)) 
        
        W = changeweight(value,X_line,Y,W,label_index,a) 
        
        s_test = X_test.shape[0] #number of data in X_test
        for j in range(s_test):
            x_test = X_test[:,line_index][j] 
            Y_testp[j] += AdaClassify(a,x_best,x_test)
        
            y_dot = Y_testp[j] *Y_test[j]
            if (y_dot <= 0):      
                n += 1 
        error_rate.append(n/float(s_test)) #error rate of every iteration
        print error_rate[i]
                    
    return error_rate

#Adaboost classify(combination of weak classifier)
def AdaClassify(alpha,value_best,x_test):

    value, label_index ,line_index = value_best[0],value_best[1][0],value_best[1][1]  #value, label_index ,line_index = best value, l/r, X1/X2 
    y_testp = lineclassify(value,x_test,label_index)*alpha #adaboost classifier
               
    return y_testp

f = pd.read_csv('hw2_data_1.txt',sep='\t', header = None, skiprows = 1)

X = f.loc[:69,[0,1]].values
Y = f.loc[:69,[2]].values
X_test = f.loc[70:,[0,1]].values
Y_test = f.loc[70:,[2]].values

s = X.shape[0]
W = np.full((s,1),1/float(s))

iteration = 20

error_rate = Adaboost(iteration,X,Y,W,X_test,Y_test)
print error_rate[iteration-1]

x_plot = list(range(iteration))

plt.plot(x_plot, error_rate)
plt.xlabel('iteration')
plt.ylabel('error rate')
plt.show()
