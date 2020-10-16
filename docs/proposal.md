---
layout: default
title: Proposal
---

## Summary of the Project

## AI/ML Algorithms

## Evaluation Plan## Summary of the Project
Our CS 175 Minecraft project will focus on geolocation estimation 
using one photo of the Minecraft scene. Our input will be a single image 
taken by the agent at ground level and our output will be an estimation 
of the longitude and latitude of the current block the agent is standing on. 

## AI/ML Algorithms
We anticipate using various type of Convolutional Neural Network for our project.

## Evaluation Plan
### Quantitative Evaluation
We plan using mean squared error and root mean squared error 
for our quantitative evaluation. The baseline model will be guessing 
the latitude and longitude randomly. We expect our model to improve 
moderately compared to the baseline but not pinpoint accuracy as it may 
be difficult to determine where exact the agent is based on one single picture.

### Qualitative Evaluation
For qualitative analysis, we would like to evaluate the model based on 
certain landmarks. That is, whether the agent could estimate a location 
where a landmark could be seen from both the original coordinates and the 
predicted coordinates. 


