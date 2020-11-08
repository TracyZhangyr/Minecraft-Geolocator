---
layout: default
title: Status
---


# Placeholder for Youtube Video

# Project Summary

## Summary

In our CS 175 Project, we are interested in the problem of image regression. Given an image of size 640x360x3 in a custom made map of Washington D.C., predict the coordinates of where that image was taken in Minecraft. That is, the result is two coordinates x and z. We solve through gathering image data and their corresponding coordinates to train Convolutional Neural Networks and Multi-Layer Perceptrons. Finally, we would save the best performing neural network model and load it into an agent. This agent would pass a test image through the model, receive x and z coordinates, and walk automatically to the predicted coordinates. 

We would be gather training data from 4 directions, North, South, East, West, respectively. The model would be trained in an area 400x400 area. -200 to 200 in the x-direction and -200 to 200 in the z-direction. Pictures are taken every 10 blocks in the x-direction and the z-direction.

## Picture Summary
<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/high_level.png" width="800" />

## Changes Made

1.	Increased training area from 200x200 to 400x400
2.	Image resolution is set to 640x360x3
3.	Map changed from set survival Minecraft seed to custom made map of Washington D.C.



## What we solved in our prototype

We solved a simpler problem in our status report. The problem is limited in size and difficulty. We tuned down the training area to 200x200, and we limited the training data to 1 direction (East). 

# Approach

We created an agent called coordinatebot.py to take pictures in the designated areas. The agent took a total of 441 colored images with a resolution of 640x360 (16:9). We created 4 neural networks (3 MLP and 1 CNN) for coordinate regression. One of the MLP is our baseline model. We will give our approach to all models in the space below.

## Multi-Layer Perceptron 

For MLP, we devised 3 different types of feature space.

### Each individual pixels as features (Baseline Model)

We first converted the image to grayscale to reduce the number of features. The original image has R, G, and B channels. That is 640x360x3. We would have 691200 input features for our Multi-Layer Perceptron. That is too much! We reduced the number of input features by resizing the image and grayscaling the image. We rescaled the image to 256x144x3, and we converted the image to grayscale using the following formula. 

Y = 0.2989 R + 0.5870 G + 0.1140 B

This allows us to reduce the number of input features to 36864 as we eliminated the R, G, and B channels and scaled the image down. The MLP model we used has the specification below:

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/mlp_pixel.PNG" />

From [Tensorflow’s Website](https://www.tensorflow.org/js/guide/models_and_layers): Note the None values in the output shapes of the layers: a reminder that the model expects the input to have a batch size as the outermost dimension, which in this case can be flexible due to the null value.

The first 3 dense layers have Rectified Linear Unit activation function. The last dense layer has a linear activation function. The dropout layer has a dropout rate of 0.05. The loss function is mean square error and the optimizer is Adam. We trained the model with 40 epoches and with a batch_size of 3.


### Oriented FAST and Rotated BRIEF (ORB)

FAST stands for Features from Accelerated Segment Test and BRIEF stands for Binary Robust Independent Elementary Features. FAST is used for corner detection and BRIEF is a feature descriptor. A feature descriptor is an algorithm that takes an image and output feature vectors of that image. We believe that images that are in close proximity to another will have similar feature vectors.

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