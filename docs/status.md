---
layout: default
title: Status
---


# Placeholder for Youtube Video

# Project Summary

## Summary

In our CS 175 Project, we are interested in the problem of image regression. Given an image of size 640x360x3 in a custom made map of Washington D.C., predict the coordinates of where that image was taken in Minecraft. That is, the result is two coordinates x and z. We solve through gathering image data and their corresponding coordinates to train Convolutional Neural Networks and Multi-Layer Perceptrons. Finally, we would save the best performing neural network model and load it into an agent. This agent would pass a test image through the model, receive x and z coordinates, and walk automatically to the predicted coordinates. 

We would be gather training data from 4 directions, North, South, East and West. The model would be trained in an area 200x200 area. -100 to 100 in the x-direction and -100 to 100 in the z-direction. 


## Picture Summary
<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/high_level.png" width="800" />

## Changes Made

1.	Image resolution is set to 640x360x3
2.	Map changed from set survival Minecraft seed to custom made map of Washington D.C.


## What we solved in our prototype

We solved a simpler problem in our status report. The problem is limited in size and difficulty. We limited the training data to 1 direction (Positive z direction).

# Approach

We created an agent called coordinatebot.py to take pictures in the designated areas. The agent took a total of 441 colored images with a resolution of 640x360 (16:9). We created 4 neural networks (3 MLP and 1 CNN) for coordinate regression. One of the MLP is our baseline model. We will give our approach to all models in the space below.

## Multi-Layer Perceptron 

For MLP, we devised 3 different types of feature space.

### Each individual pixels as features (Baseline Model)

We first rescaled the image to 256x144x3 to reduce the number of pixels needed for training. The original image has a resolution of 640x360x3. We would have 691200 input features for our Multi-Layer Perceptron. That is too much! We reduced the number of input features by resizing the image and grayscaling the image. We, and we converted the image to grayscale using the following formula. 

Y = 0.2989 R + 0.5870 G + 0.1140 B

This allows us to reduce the number of input features to 36864 as we eliminated the R, G, and B channels and scaled the image down. The MLP model we used has the specification below:

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/mlp_pixel.PNG" />

From [Tensorflow’s Website](https://www.tensorflow.org/js/guide/models_and_layers): Note the None values in the output shapes of the layers: a reminder that the model expects the input to have a batch size as the outermost dimension, which in this case can be flexible due to the null value.

The first 3 dense layers have Rectified Linear Unit activation function. The last dense layer has a linear activation function. The dropout layer has a dropout rate of 0.05. The loss function is mean square error and the optimizer is Adam. We trained the model with 40 epoches and with a batch_size of 3.


### Oriented FAST and Rotated BRIEF (ORB)

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/orb.PNG" />

FAST stands for Features from Accelerated Segment Test and BRIEF stands for Binary Robust Independent Elementary Features. FAST is used for corner detection and BRIEF is a feature descriptor. A feature descriptor is an algorithm that takes an image and output feature vectors of that image. The blue circles on the image above are points that the algorithm believe are interesting. We believe that images that are in close proximity to another will have similar feature vectors. 

We used Opencv’s ORB function to detect and compute the descriptors of each individual image. A total of 100 features are selected from each individual image. We flattened the feature vectors for each image and used them as the input for our MLP. The MLP model we used has the specification below:

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/mlp_orb.PNG" />

The first 3 dense layers have Rectified Linear Unit activation function. The last dense layer has a linear activation function. The dropout layer has a dropout rate of 0.01. The loss function is mean square error and the optimizer is Adam. We trained the model with 20 epoches and with a batch_size of 3.

### Landmark ORB

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/ggb.jpg" width='450'/>

This idea came from how humans recognize a location. Given the example of the picture above, we know that the picture is taken in San Francisco because we recognize the Golden Gate Bridge. We tried to simulate this idea with our dataset.

We uniformed chose 40 pictures from our 441 dataset and set them as landmarks. We ran ORB algorithm on these 40 landmark pictures and their feature vector are used as landmark features. We then computed the feature vector for the rest of the dataset and ran a brute force feature matching algorithm provided by OpenCV to those 40 landmark pictures. The amount of key points matched and their corresponding match distance are used as feature space for MLP. The MLP model we used has the specification below:

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/mlp_landmark.PNG" />

The first 3 dense layers have Rectified Linear Unit activation function. The last dense layer has a linear activation function. The dropout layer has a dropout rate of 0.01. The loss function is mean square error and the optimizer is Adam. We trained the model with 50 epoches and with a batch_size of 3.

## Convolutional Neural Network

While we have to devise feature extraction methods for MLP, CNN, on the other side, does need manual feature extraction. The convolution layers followed by max-pooling layers serve as the feature extraction steps for CNN. Then we apply a hidden layer to distinguish those features and output two continuous variables. We load the image and grayscale the image to reduce computational cost while retaining the majority of image features. We then feed the grayscaled image through the CNN model with the specified architecture below.

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/cnn.PNG" />

All the Conv2D layers have Rectified Linear Unit activation function with SAME padding. The Dropout layer has a dropout rate of 0.2, and the Dense layer has a Rectified Linear Unit activation function. The output layer does not have an explicit activation function; it defaults to a linear activation function. We trained the model with 20 epoches and with a batch_size of 3. 

# Evaluation

## Quantitative Evaluation

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/equation.PNG" width='500' />

We will be evaluating our models quantitatively using mean squared error. Mean square error measures the average of the squares of the difference between the estimated values and the actual value. In the context of this project, mean squared error is applied to the estimated coordinates (x and z) of where an image was taken and the true coordinates of where that image was taken. Note that the MSE formula above is not the average Euclidean distance error. However, the formula is proportional to the average Euclidean distance error. In general, the lower the MSE is, the close our estimated coordinates are to the true coordinates. We performed 5 fold cross-validation on all of our models and got the average mean squared error for the test samples and the train samples. We are not only interested in the test accuracy but also interested in the training accuracy. The results are presented in the table below.

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/table.PNG" width='500' />

We could see from the MSE table above that MLP ORB Features model outperforms the baseline model (MLP pixel) by a large margin, both on Train MSE and Test MSE. MLP ORB Features model outperformed baseline model by a MSE of 3302 on training data and a MSE of 1455 on the testing data. To give a more concrete idea, given a train image, MLP ORB Features model would, on average, predict a coordinate that is 16.5 blocks away from the true coordinates in x-axis and z-axis. Given a test image, MLP ORB Features model would, on average, predict a coordinate that is 56 blocks away from the true coordinates in x-axis and z-axis. In addition, we could also observe from the MSE table that our CNN model is performing better than the baseline. Eventhough the model is overfitting, CNN still has a lower MSE compared to the baseline. The high Test MSE is within our expectation because we used a real life environment. Real life environments are complexed and highly subjected to change. Even 10 blocks away could have drastically different image features. There are also noisy images that decrease model accuracy. Thus, we believe that the actual Test MSE for the models should be slightly lower if noisy images are removed.

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/Status%20Report-MSE%20Low.png" />

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/Status%20Report-MSE%20High.png" />

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/Status%20Report-Page-5.png" />

## Qualitative Evaluation

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/Status%20Report-Picture.png" />

We loaded the models onto our result checker. We gave each model the same test picture as our quantitative evaluation, and the results are shown above. The test image consists of the right-side of the Whitehouse replication, a carlike object as well as some forest terrain on the right side. While models have high test MSE numbers, as indicated in the quantitative evaluation, we could see all the predictions retain some test image features. CNN predicted an enlarged image of the entrance to the Whitehouse. MLP Pixel, MLP ORB, and MLP Landmark seem to be predicting more of the forest. However, none of the predicted images have the blue carlike object in the test image. We also created a Cartesian coordinate graph with the test image centered at (0, 0). The coordinates of the model predictions have been adjusted to the relative center and the direction the agent is facing.  

# Remaining Goals and Challenges

## Goals 

Now that we have demonstrated that performing coordinate regression prediction on a single image is a viable option, we would like to upscale our project by incorporating several images of a single coordinate. Our prototype only used images from 1 direction rather than 4 directions. Hopefully, the additional images will provide more information, which may decrease MSE. We have also shown in our status report that a simple CNN model outperforms all MLP models we have created. We would like to investigate the performance of some CNN architectures such as AlexNet, VGG, and LeNet-5 (the specific CNN architecture to be implemented is subjected to change) in our final report.

## Challenges

We anticipate 2 challenges: Machine Learning with multiple images of the same coordinate, and the time/computational cost of training large CNN architectures. We will address the problem of having to learn from multiple images of the same coordinate first. We will take 4 images (North, South, East, and West) at a 90 degree field of view in the same position and try to develop a machine learning model that performs multi-view learning. One potential issue is figuring out how we should adjust our models with the new data format. For example, the input shape for multi-view image data would be different from single image data (the one we did in this status report). We would solve this problem through reading some papers or looking at some models that implement some form of multi-view learning. The second challenge is the time and computational cost for training CNN architectures. We created a simple CNN for our status report, and it took around 4 hours to ran 5 fold cross-validation. We anticipating building larger networks with a larger dataset for our final report meaning the training time will get longer. One solution is to grayscale the image to reduce image dimensions. This problem is tricky because performing dimensionality reduction will cause the image to lose some features that may be relevant to regression. We would have to find the balance between feature preservation and dimensionality reduction.


# Resources Used

[Project Malmo](https://www.microsoft.com/en-us/research/project/project-malmo/)

[TensorFlow](https://www.tensorflow.org/)

[OpenCV](https://opencv.org/)

[Feature detection and matching with OpenCV](https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590)

[Regression to the mean and its implications](https://towardsdatascience.com/regression-to-the-mean-and-its-implications-648660c9bf76)

[Feature extraction from images](https://www.kaggle.com/lorinc/feature-extraction-from-images)

[Deep Learning Models for Multi-Output Regression](https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/)

[XML Schema Documentation](https://microsoft.github.io/malmo/0.14.0/Schemas/Mission.html)

[scikit-learn](https://scikit-learn.org/stable/)

[Basic regression: Predict fuel efficiency](https://www.tensorflow.org/tutorials/keras/regression)

[PRODIGIOUS WASHINGTON (Minecraft Map)](https://www.minecraftmaps.com/city-maps/prodigious-washington)