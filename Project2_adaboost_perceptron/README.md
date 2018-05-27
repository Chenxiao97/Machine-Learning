Bios/CS 534 Project 2

Please use Python for programming and plotting. Please compile all result into a single PDF file and submit via Emory Canvas. Code must be appended at the end of the file.

For the first two problems, please do your own programming, and do not use existing packages.

(1) Download the data from http://web1.sph.emory.edu/users/tyu8/534/. The files is “hw2_data_1.txt”. The data contains two X variables and one Y variable (two classes). Use rows 1-70 as training data, and use the remaining rows as testing data.

Write a function of perceptron. Using initial weights of all “1”s, and a learning rate of 1, run the perception on the training data. Conduct prediction on the testing data, and report error rate.

(2) Write a function of Adaboost. The base learner should be a cutoff on one of the axes. The threshold can be found by greedy search over all possible cutoffs.

Using the same data as in question (1), Run the function on the training data with 3, 5, 10, and 20 iterations. For each of the iteration settings, conduct prediction on the testing data, and report error rate.

For the following problems, you can use existing packages to complete the task.

(3) Download the data from http://web1.sph.emory.edu/users/tyu8/534/. The file is “hw2_data_2.txt”. The data contains twenty X variables and one Y variable (two classes). Use rows 1-700 as training data, and use the remaining rows as testing data.

Run Support Vector Machine with three different kernels: Radial, polynomial, and sigmoid. Keep every other parameter default, except the following:

For radial kernel, run a grid search for the best gamma parameter. Use 10-fold cross-validation to select the best gamma parameter. Plot the cross-validation error rate v.s. gamma parameters. Fit the final model using the selected gamma, and conduct prediction on the testing data. Report testing error rate.

For sigmoid kernel, conduct the same procedure as for the radial kernel. Use 10-fold cross-validation to select the best gamma parameter. Plot the cross-validation error rate v.s. gamma parameters. Fit the final model using the selected gamma, and conduct prediction on the testing data. Report testing error rate.

For the polynomial kernel, tune the degree parameter using 10 fold cross validation. Plot the cross-validation error rate v.s. degree parameters. Fit the final model using the selected degree, and conduct prediction on the testing data. Report testing error rate.

(4) Run the Random Forest on the training data. Keeping all other parameter at default, fit the model using different number of trees (10, 20, 30,……, 500). Plot the OOB error rate v.s. the number of trees. Select the number of trees based on when the OOB error rate first stabilizes (close enough to the error rate at 500 trees, with a difference within 0.2%).

Using the selected number of trees, fit the model, and conduct prediction on the testing data. Rank the variables based on their importance.

(5) Run gradient boosting with deviance loss on the training data. Predict the testing data. Report the testing data error rate. Report variable importance.

(6) Rum MARS on the training data. You can treat the outcome variable as continuous. Predict the testing data. If y is treated as continuous, please dichotomize the predicted outcome at the median. Report the testing data error rate.
