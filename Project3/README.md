Project 3 Building image classifier for MNIST dataset 
using Tensorflow

1.	Introduction

Tensorflow is a popular open source framework for building deep learning applications. In this project, you will use tensorflow to build an image classifier for MNIST dataset and save the model you trained for submission. Tensorflow eases the needs for building backpropagation part, which requires you to differentiate the objective function with respect to the weights in each layer. The only thing you need to focus on is the forward pass of your network, in other words, the way your input tensor (data) flows in the predictive model.
If  you would like to know more theoretical knowledge about convolutional neural network, I recommend you to watch the following video course: 
Lecture 5 of CS231n from Standord Univerisity
If  you would like to know more about how to tune hyper-parameters of a neural network, I recommend you to watch the following video courses: 
Lecture 6 of CS231n from Stanford University and 
Lecture 7 of CS231n from Stanford University
These videos may help you gain more insights about this project.

2.	Software requirement
Tensorflow 1.6 (latest stable version)
Official API for 1.6 (https://www.tensorflow.org/api_docs/python/)

Please do NOT use other versions of tensorflow since many methods in tensorflow are different in different versions. The latest version would ensure the official tutorial mentioned below works perfectly.

Refer to the official API whenever you meet problems using tensorflow.

3.	Building conv-net using tf.layers
Building a convolution neural network is not painful with tensorflow. You can firstly refer to the official tutorial (https://www.tensorflow.org/tutorials/layers), which builds a conv-net with 2 conv layers, 2 pooling layers and 2 fully connected layers, to know how to build a convolutional neural network from scratch with tensorflow. In fact, you can simply use tf.layers.conv2d() to build a conv layer and tf.layers.max_pooling2d() to build a pooling layer in one line of code. 
Tasks:
1)	Load the MNIST data preprocessed by tensorflow. 
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
You can access the training data and testing data by using the following code:
train_data = mnist.train.images (55000, 784)
eval_data = mnist.test.images (10000, 784)

You can then split part of the data (for example 20%) from train_data as validation data and use that part for evaluating the performance of your model when training. (Remember: the validation accuracy is only used for monitoring whether your model has a good generalization performance while training. This is not aimed for report the final performance.)
In this project using this fixed validation data is fine, since the computation load would be high if you apply 5 or 10-fold cross validation. The testing data (test_batch) is only used for reporting the performance on selected best model (IMPORTANT).

Hint: To perform periodically evaluate on the validation set, you may need to use the class tf.contrib.learn.Experiment. You can use the method continuous_train_and_eval() after you instantiate the class.
Example code:
  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=200,
      num_epochs=None,
      shuffle=True)
  valid_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": valid_data},
      y=valid_labels,
      num_epochs=1,
      shuffle=False)
  experiment = tf.contrib.learn.Experiment(mnist_classifier, train_input_fn, valid_input_fn, train_steps=5000, eval_steps=None, train_steps_per_iteration=500)
experiment.continuous_train_and_eval()

•	You can evaluate the performance of the validation set every 500 steps out of the total 5000 steps. 
•	1 step in this example means training on 200 samples (batch_size). 
•	It is suggested by the tensorflow official team that setting shuffle equals true for training set will help the model reduce the chance of overfitting.
•	num_epochs = 1 and batch_size unset will let you evaluate on the entire validation set. 
•	batch_size is also a parameter you might need to tune. A too large batch_size will increase the computation load for every backpropogation and a too small batch_size will make the gradients noisy and influence the convergence rate of the optimization algorithm.

2)	Try different structures of the conv-net and compare the performance with each other. The components you can modify but not limited to are: initial learning rate, size of the filters, number of filters, number of conv layers and pooling layers, the ways of paddings, the choices of activation functions and etc. 
3)	Try different built-in optimizers in tensorflow (https://www.tensorflow.org/api_guides/python/train#Optimizers). In the original tutorial for classifying MNIST, the author used steepest gradient descent. Firstly read related materials to understand the mathematical mechanisms of momentum optimizer and adam optimizer (most popular for now) and then observe the convergence rate compared to steepest gradient descent. (Hint: Observe the training loss with respect to number of steps). 
4)	Add tensorboard part to your model. Tensorboard is a powerful tool, which helps you monitor how your training process goes on. It is a good way to visualize your model (computation graph you generated with tensorflow) and help you tune a good combination of hyper-parameters. You can even visualize how the weights of your model change with respect to training epochs. Read the tutorial below and watch the great video on this page for more information on adding tensorboard to your code. (https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)
Hint: In this project, you may not need to initialize tf.summary.FileWriter by your own since tf.estimator.Estimator will automatically help you write the summary you want to record into the event file (See this discussion). You may just add tf.summary.scalar('loss', loss) after the loss computed in the original code and then you can view the result in the tensorboard after the training finishes.
To visualize the tensorboard event file on the browser:
•	Tensorboard --logdir path_to_your_model_dir --port=8008
•	Then type http://localhost:8008 in your the browser 
5)	Write up a report about how you did the experiment in step 1) – 4). Report the performance on testing data based on the best model you have tuned (the structure of the model, the regularization methods you applied to prevent overfitting and any other components you added to make your model better and etc.) and attach the training plots generated by tensorboard to show how the training process goes. You may need to include how training and validation loss changes, how training and validation accuracy changes and the computation graph you generated (in the GRAPH part of the tensorboard log file). 

Sample computation graph generated by tensorboard for the MNIST tutorial:

4.	Submission
1)	The report 
2)	The code
3)	The tensorboard event files you generated (events.* files in your model directory) – best run only
Note: you do not need to submit the model.cpkt* files. These files are used for restoring your model and too large to uploaded in canvas.

