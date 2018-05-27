Download the data from http://web1.sph.emory.edu/users/tyu8/534/. The files are “HW_1_training.txt “ and “HW_1_testing.txt”. The data contains two X variables and one Y variable (two classes).

(1) Assuming normality of each class, calculate the mean vector and covariance matrix of each class based on the training data. Construct a Bayesian decision boundary for the following two scenarios:

(a) equal prior;

(b) prior calculated from the data.

Plot each of the boundary on the scatter plots of the training data. Calculate the classification error rate on the testing data.

Note: Please show your work of computing the boundary. Do NOT use the QDA function.

(2) Using the training data, find the class-specific densities using kernel density estimator. Using the Bayesian decision rule with equal prior, predict the class labels on the testing data. Calculate the error rate. Repeat the procedure using three kernel sizes:

If you use sklearn, please use symmetric kernels with bandwidth 10, 1, 0.1;

If you use scipy, which allows non-symmetric kernel, please use

(a) 0.1×covariance matrix of the class-specific training data,

(b) 1×covariance matrix of the class-specific training data.

(c) 10× covariance matrix of the class-specific training data.

Plot the testing data and color the misclassified points differently.

(3) Code a function for the K-nearest neighbor classifier. Please do not use the built-in KNN classifier.

Classify the data points in the testing data using three settings: (a) K=1, (b) K=5, (c) K=10.

For each setting, consider class “1” as disease cases and “0” as healthy controls. Calculate the sensitivity, specificity, and false discovery rate. Plot the testing data. Use different shape to show the true class labels, and color the misclassified points with different colors.
