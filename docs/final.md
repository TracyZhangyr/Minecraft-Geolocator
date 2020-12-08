---
layout: default
title: Final Report
---


## Video 

Place Holder for Video

## Project Summary

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img_final/CS175%20Final%20Diagrams-Intro%20picture.png" width='600' />

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

https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img_final/CS175%20Final%20Diagrams-Shape%20explained.png

We convert input images to grayscale and normalize image data to the range [0,1] to reduce computational cost while retaining most image features.

### Models Created