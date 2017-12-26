# **Traffic Sign Recognition** 

## Constructing a neural network using Tensorflow for the classification of  

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./training_data_dist.png "Visualization"
[image1a]: ./training_data_dist_new.png "Visualization1"
[image1b]: ./class_accuracies.jpg "Grayscaling"
[image2]: ./examples/grayscale.jpg "Grayscaling1"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples_test/13.jpg "Traffic Sign 1"
[image5]: ./examples_test/14.jpg "Traffic Sign 2"
[image6]: ./examples_test/17.jpg "Traffic Sign 3"
[image7]: ./examples_test/27.jpg "Traffic Sign 4"
[image8]: ./examples_test/3.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ShujaKhalid/Computer-Vision-TrafficSignClassifier)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram representing the frequency of signs in the training data set. The training dataset is thus not uniform and requires additional samples for proper training. An analysis of the accuracies for the various classes is needed to determine which of the classes the classifier is most confused about. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to convert the images to graysccale because the LeNet paper indicates that the inclusion of RGB images during training did not result in significant performance gains (http://yann.lecun.com/exdb/lenet/). 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

The next step was to create additional images as the visualization of the data clearly shows that there aren't enough images to properly train the model. A number of images were thus modified to create new images. A summary of the images that were modified and how they were modified is included in [signnames.csv] (https://github.com/ShujaKhalid/Computer-Vision-TrafficSignClassifier/blob/master/signnames.csv)

The resulting data distribution is presented below:

![alt text][image1a]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution      	| 5x5 kernel with a 1x1 stride 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution    | 5x5 kernel with a 1x1 stride  									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Flatten | 
| Fully connected		| etc.        									|
| RELU					|												|
| Fully connected		| etc.        									|
| RELU					|												|
| Fully connected		| etc.        									|
| RELU					|												|
| Softmax				| etc.        									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer, a batch size of 128 and 500 epochs. The learning rate was chosen to be 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.99
* validation set accuracy of 0.94
* test set accuracy of 0.93

The LeNet architecture was chosen for the analysis and in order to gain better performance, the hyperparameters were tuned. The aforementioned LeNet structure has shown to provide very good results in identifying traffic signs and digits (http://yann.lecun.com/exdb/lenet/).

The chosen model is the right model for the application because it produces encouraging results despite the lack of data for a number of classes, as can be seen in the image below:

[image1b]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Some of the images extracted from the web included watermarks and terrible lighting. The images also had to be cropped to the required dimensions (32x32) in order to be fed into the model.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the of the Ipython notebook, titled, Output Top 5 Softmax Probabilities For Each Image Found on the Web.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability     |     Prediction	        					| 
|:--------------------:|:---------------------------------------------:| 


For the second image ... 

| Probability   	 |     Prediction	        					| 
|:--------------------:|:---------------------------------------------:| 


For the third image ...

| Probability    	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 


For the fourth image ...

| Probability    	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 


For thefifth image ...

| Probability    	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 

