---
layout: default
title: Final Report
---


## Video 

Place Holder for Video

## Project Summary

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img_final/CS175%20Final%20Diagrams-Intro%20picture.png" width='700' />

In our CS 175 Project, we are interested in the problem of image regression. Given 4 images of size 640x360x3 in a custom made map of Washington D.C., predict the coordinates of where that image was taken in Minecraft. That is, the result is two coordinates x, and z. We solve through gathering image data and their corresponding coordinates to train Convolutional Neural Networks. 

We would be gather training data from 4 directions, North, South, East, and West, for a single pair of coordinates. The model would be trained in an area 200x200 area. -30 to 130 in the x-direction and 0 to 200 in the z-direction. Images will be taken every 10 blocks in the x direction and every 10 blocks in the z direction.  

Our goal for this project is to train convolutional neural networks to perform better than the baseline. We expect a successful model to have less than 2700 MSE. This roughly equates to predicting a coordinate with a 36 block difference from the ground truth in the x axis and the z axis. 

We would like to specify some specific language we would be using from now on. We will not be using North, South, East, and West to describe the 4 different views. Instead, we would be using positive x, positive z, negative x, negative z to describe which way the agent is facing. 

### Why do we need ML/AI for this problem

This problem is non-trivial because we are not sampling every coordinate in the training/testing space. If we were to sample images from every pair of coordinates, we could just use a brute force pixel RGB color matching program. In our case, we are sampling images every 10 blocks in the x axis and z axis. We need AI for this problem because it is difficult to identify and extract image features manually and it is difficult to infer the distance between 2 different images taken. Algorithms that detect image features such as ORB (Oriented FAST and Rotated BRIEF) are good with detecting image features and could identify if 2 images are similar. However, ORB algorithm lacks the ability to infer how far 2 images are taken. This is critical for our task as we need to be able to not only know that 2 images are similar but also know how far apart the 2 images are. Thus, we need ML algorithms for this task. 

We are also not using a simple map. We are using a complex map with objects such as buildings, roads, cars, trees, etc. In addition, there are also an elevation difference between the images taken. This makes the problem difficult as the environment simulates a real environment rather than a simplified testing environment.

## Approach

This section will be divided into 3 smaller sections: Data Collection, Data Dimensionality Reduction, as well as Models.

### Data Collection

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img_final/CS175%20Final%20Diagrams-Data%20gathering.png" />

#### Image Dimension

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img_final/CS175%20Final%20Diagrams-Shape%20explained.png" />

### Data Dimensionality Reduction

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img_final/CS175%20Final%20Diagrams-Data%20Preprocessing.png" />

We convert input images to grayscale and normalize image data to the range [0,1] to reduce computational cost while retaining most image features.

### Models Created

#### Predict Center (Baseline)

This model always predicts the center of the training dataset regardless of the image input. This is a statistical phenomenon known as regression to the mean. When the input features are not correlated with the target values, the optimizer will start to predict towards the center to minimize the loss. The center for our training dataset is [30,100]. The model will always output [30, 100] as its prediction, regardless of what image is feed into it. The advantage of this model is that it does not require machine learning. It needs very minimal time to train because it only needs to average the training coordinates to get the center for the training dataset. The disadvantage of this model is that it is not very accurate. It will always predict the center regardless of how far or how close the image was taken relative to the center. This model is not very practical as one would want to know their current position. Thus, this model is our baseline model.

#### LeNet-5 Individual

This model is based on the LeNet-5 model. It is not the exact LeNet-5 model because we have made modifications to it. This model extracts features from the 4 different images individually. Each direction (+z, -z, +x, -z) has its own convolution layers to extract features, which are then concatenated and feed into a feed-forward multi-layer perceptron. One advantage of this model is that due to its unique architecture of extracting features from the 4 directions individually, the model is able to use Conv2D layers. Recall the shape of our training data is (441, 4, 360, 640, 1). The data has 5 dimensions, and Conv2D layers only accept 4 dimensions, namely (data size, height, width, channels). Using 4 different Conv2D layers to extract image features from the 4 directions allows us to reduce the data dimension to 4 rather than 5. Conv2D has an advantage over Conv3D as it trains faster. This gives us more time to tune the model and adjust it. One disadvantage of this model is that it is not able to recognize what image feature is from what direction. Sometimes images from different coordinates have similar features, but these similar features are captured in a different direction. This will cause the model to think that these 2 images are taken near each other due to their similar image features. The model is unaware of the fact that similar image features are from different directions causing high MSE loss. 

We will present 2 pictures below: a high-level picture of the model’s architecture and detailed configurations.

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img_final/CS175%20Final%20Diagrams-lenet%20individual.png" />

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img_final/indivudal%20model.PNG" />

The first convolutional layer has filters = 8, kernel_size = (3, 3), activation = relu, padding = same. The second convolutional layer has filters = 16, kernel_size = (3, 3), activation = relu, padding = same. The first dense layer has units = 512, with linear activation as default and the output layer has an output size of 2 with linear activation function. We fit the training data with batch_size = 10 and epochs = 20. To avoid overfitting the data, we decide to add dropout layers and stop early. We programed a callback function that will stop training once the loss fell below 1700. We got this number after trial and error. It was determined that the optimal test and validation values occurs when the loss is in the range of 1500 to 1800. 
