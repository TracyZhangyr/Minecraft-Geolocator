---
layout: default
title: Status
---


# Placeholder for Youtube Video

# Project Summary

## Summary

In our CS 175 Project, we are interested in the problem of image regression. Given an image of size 640x360x3 in a custom made map of Washington D.C., predict the coordinates of where that image was taken in Minecraft. That is, the result is two coordinates x and z. We solve through gathering image data and their corresponding coordinates to train a Convolutional Neural Network. Pixels and their corresponding colors would be features and the target would be two continuously variables, x-coordinate and z-coordinate.

We would be gather training data from 4 directions, North, South, East, West, respectively. The model would be trained in an area 400x400 area. -200 to 200 in the x-direction and -200 to 200 in the z-direction. Pictures are taken every 5 blocks in the x-direction and the z-direction.

## Changes Made

1.	Increased training area from 200x200 to 400x400
2.	Image resolution is set to 640x360x3
3.	Map changed from set survival Minecraft seed to custom made map of Washington D.C.
4.	Change pictures taken every 10 blocks in the x-direction and z-direction to every 5 blocks in the x-direction and z-direction


## What we solved in our prototype

We solved a simpler problem in our status report. The problem is limited in size and difficulty. We tuned down the training area to 200x200, and we limited the training data to 1 direction (East). 

# Approach

## Multi-Layer Perceptron (Baseline Model)

## Convolutional Neural Network