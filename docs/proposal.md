---
layout: default
title: Proposal
---

## Summary of the Project
Our CS 175 Minecraft project will focus on geolocation estimation 
using one photo from Minecraft. Our input will be a single image 
taken by the agent at ground level and our output will be an estimation 
of the longitude and latitude of the current block the agent is standing on.
We got our inspiration from the game [GeoGessr](https://www.geoguessr.com/), a game
where humans guess the location of the world given google map images. We are wondering
how well machines could determine geolocation given an image. Our project could extend
to real world applications such as automatic geolocation tagging for social media sites
like instagram or facebook. 

![overview pic](https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/general_overview.PNG)

## AI/ML Algorithms
We anticipate building few different Convolutional Neural Network architectures for our project.


## Evaluation Plan
### Quantitative Evaluation
We will be evluating the errors between the predicted location and the
true location. We plan to use mean squared error and root mean squared error on the
longitude and latitude. The baseline model will be guessing 
the latitude and longitude randomly. We expect our model to improve 
moderately compared to the baseline but not pinpoint accuracy as it may 
be difficult to determine where exact the agent is with no other
information but only one single picture.


### Qualitative Evaluation
For qualitative analysis, we would like to evaluate the model based on 
certain landmarks. That is, whether the agent could estimate a location 
where a landmark could be seen from both the original coordinates and the 
predicted coordinates. 


