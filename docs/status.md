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

We created an agent called coordinatebot.py to take pictures in the designated areas. The agent took a total of 441 colored images with a resolution of 640x360 (16:9). We created some neural networks (MLP and CNN) for coordinate regression. One of the MLP is our baseline model, and we the other models we created against the baseline. We will give our approach to all models in the space below.

## Multi-Layer Perceptron 

For MLP, we devised 3 different types of feature space.

### Each individual pixels as features (Baseline)

We first converted the image to grayscale to reduce the number of features. The original image has R, G, and B channels. That is 640x360x3. We would have 691200 input features for our Multi-Layer Perceptron. That is too much! We reduced the number of input features by resizing the image and grayscaling the image. We rescaled the image to 256x144x3, and we converted the image to grayscale using the following formula. 

Y = 0.2989 R + 0.5870 G + 0.1140 B

This allows us to reduce the number of input features to 36864 as we eliminated the R, G, and B channels and scaled the image down. The MLP model we used has the specification below:

### Oriented FAST and Rotated BRIEF (ORB)



## Convolutional Neural Network