---
layout: default
title: Status
---


<iframe width="560" height="315" src="https://www.youtube.com/embed/AOWhdx-pizg" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Project Summary

## Summary

In our CS 175 Project, we are interested in the problem of image regression. Given an image of size 640x360x3 in a custom made map of Washington D.C., we predict the coordinates of where that image was taken in Minecraft. That is, the result is two coordinates x and z. We solve through gathering image data and their corresponding coordinates to train Convolutional Neural Networks and Multi-Layer Perceptron. Finally, we would save the best performing neural network model and load it into an agent. This agent would pass a test image through the model, receive x and z coordinates, and walk automatically to the predicted coordinates.

We would be gathering training data from 4 directions, North, South, East, and West, for a single pair of coordinates. The model would be trained in a 200x200 area: -70 to 130 in the x-direction and 0 to 200 in the z-direction. Images will be taken every 10 blocks in the x-direction and every 10 blocks in the z-direction.


## Picture Summary
<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/high_level.png" width="800" />

## Changes Made

1.	Image resolution is set to 640x360x3
2.	Map changed from set survival Minecraft seed to custom made map of Washington D.C.


## What we solved in our prototype

We solved a simpler problem in our status report. The problem is limited in size and difficulty. We limited the training data to 1 direction (Positive z direction).

# Approach

We created an agent called coordinatebot.py to take pictures in the designated areas. The agent took a total of 441 colored images with a resolution of 640x360 (16:9). We created 4 neural networks (3 MLP and 1 CNN) for coordinate regression. One of the MLP is our baseline model. We will give our approach to all models in the space below.

## Multi-Layer Perceptron (MLP)

For MLP, we devised 3 different types of feature space.

### Each individual pixels as features (Baseline Model)

We first rescaled the image to 256x144x3 to reduce the number of pixels needed for training. The original image has a resolution of 640x360x3. We would have 691200 input features for our Multi-Layer Perceptron. That is too much! We reduced the number of input features by resizing the image and converting the image to grayscale. We averaged together the three-color channels (R, G, B) by using the following formula. 

Y = 0.2989 R + 0.5870 G + 0.1140 B

This allows us to reduce the number of input features to 36864 as we eliminated the R, G, and B channels and scaled the image down. The MLP model we used has the specification below:

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/mlp_pixel.PNG" />

From [Tensorflow’s Website](https://www.tensorflow.org/js/guide/models_and_layers): Note the None values in the output shapes of the layers: a reminder that the model expects the input to have a batch size as the outermost dimension, which in this case can be flexible due to the null value.

The first 3 dense layers have a Rectified Linear Unit activation function. The last dense layer has a linear activation function. The dropout layer has a dropout rate of 0.05. The loss function is mean square error, and the optimizer is Adam. We trained the model with 40 epochs and with a batch_size of 3.


### Oriented FAST and Rotated BRIEF (ORB)

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/orb.PNG" />

FAST stands for Features from Accelerated Segment Test and BRIEF stands for Binary Robust Independent Elementary Features. FAST is used for corner detection and BRIEF is a feature descriptor. A feature descriptor is an algorithm that takes an image and output feature vectors of that image. The cyan circles on the image above are points that the algorithm believe are interesting. We believe that images that are in close proximity to another will have similar feature vectors. 

We used Opencv’s ORB function to detect and compute the descriptors of each individual image. A total of 100 features are selected from each individual image. We flattened the feature vectors for each image and used them as the input for our MLP. The MLP model we used has the specification below:

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/mlp_orb.PNG" />

The first 3 dense layers have a Rectified Linear Unit activation function. The last dense layer has a linear activation function. The dropout layer has a dropout rate of 0.01. The loss function is mean square error, and the optimizer is Adam. We trained the model with 20 epochs and with a batch_size of 3.

### Landmark ORB

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/ggb.jpg" width='450'/>

This idea came from how humans recognize a location. Given the example of the picture above, we know that the picture is taken in San Francisco because we recognize the Golden Gate Bridge. We tried to simulate this idea with our dataset.

We uniformly chose 40 pictures from our 441 dataset and set them as landmarks. We ran ORB algorithm on these 40 landmark pictures and their feature vector are used as landmark features. We then computed the feature vector for the rest of the dataset and ran a brute force feature matching algorithm provided by OpenCV to those 40 landmark pictures. The amount of key points matched and their corresponding match distance are used as feature space for MLP. The MLP model we used has the specification below:

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/mlp_landmark.PNG" />

The first 3 dense layers have a Rectified Linear Unit activation function. The last dense layer has a linear activation function. The dropout layer has a dropout rate of 0.01. The loss function is mean square error, and the optimizer is Adam. We trained the model with 50 epochs and with a batch_size of 3.

## Convolutional Neural Network (CNN)

While we have to devise feature extraction methods for MLP, CNN, on the other side, does not need manual feature extraction. The convolution layers followed by max-pooling layers serve as the feature extraction part on CNN. In this part, CNN could reduce the number of parameters by local connectivity, weights sharing, and down sampling. Then we apply fully connected layers to distinguish those features and generate the predictions of coordinates, which finally output two continuous variables. We convert input images to grayscale and normalize image data to the range [0,1] to reduce computational cost while retaining most image features. We then feed the image data through the CNN model with the specified architecture below.

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/cnn.PNG" />

We choose 2D convolution layers for our 2D input image data. All three Conv2D layers have the Rectified Linear Unit activation function with the “same” padding but different numbers of filters of 16, 32, and 64. We add a MaxPooling2D layer followed by each Conv2D layer. After the above feature extraction part, we use a Dropout layer with a dropout rate of 0.2 and a Flatten layer. Then we construct fully connected layers. The first layer is a Dense layer with the units of 128 and the Rectified Linear Unit activation function. The second (output) layer is a Dense layer with the units of 2, which corresponds to the size of coordinates, and a default linear activation function. We trained the model with 20 epochs and with a batch_size of 3.


# Evaluation

## Quantitative Evaluation

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/equation.PNG" width='500' />

We will be evaluating our models quantitatively using mean squared error. Mean square error measures the average of the squares of the difference between the estimated values and the actual value. In the context of this project, mean squared error is applied to the estimated coordinates (x and z) of where an image was taken and the true coordinates of where that image was taken. Note that the MSE formula above is not the average Euclidean distance error. However, the formula is proportional to the average Euclidean distance error. In general, the lower the MSE is, the close our estimated coordinates are to the true coordinates. We performed 5 fold cross-validation on all of our models and got the average mean squared error for the test samples and the train samples. We are not only interested in the test accuracy but also interested in the training accuracy. The results are presented in the table below.

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/table.PNG" width='500' />

From the MSE table above, we could see that the MLP ORB Features model outperforms the baseline model (MLP pixel) by a large margin, both on Train MSE and Test MSE. MLP ORB Features model outperformed the baseline model by an MSE of 275 on training data and an MSE of 3194 on the testing data. For a more concrete idea, given a training image, the MLP ORB Features model would, on average, predict a coordinate that is 16.5 blocks away from the true coordinates in the x-axis and z-axis. Given a test image, the MLP ORB Features model would, on average, predict a coordinate that is 56 blocks away from the true coordinates in the x-axis and z-axis. In addition, we could also observe from the MSE table that our CNN model is performing better than the baseline. Even though the model is overfitting, CNN still has a lower MSE compared to the baseline. The high Test MSE is within our expectation because we used a real-life environment. Real-life environments are complexed and highly subjected to change. Even 10 blocks away could have drastically different image features. There are also noisy images that decrease model accuracy. Thus, we believe that the actual Test MSE for the models should be slightly lower if noisy images are removed.

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/Status%20Report-MSE%20Low.png" />

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/Status%20Report-MSE%20High.png" />

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/Status%20Report-Page-5.png" />

## Qualitative Evaluation

<img src="https://raw.githubusercontent.com/alaister123/Geolocator/main/docs/img/Status%20Report-Picture.png" />

We loaded the models onto our result checker. We gave each model the same test picture as our quantitative evaluation, and the results are shown above. The test image consists of the right-side of the Whitehouse replication, a carlike object as well as some forest terrain on the right side. While models have high test MSE numbers, as indicated in the quantitative evaluation, we could see all the predictions retain some test image features. CNN predicted an enlarged image of the entrance to the Whitehouse. MLP Pixel, MLP ORB, and MLP Landmark seem to be predicting more of the forest. However, none of the predicted images have the blue carlike object in the test image. We also created a Cartesian coordinate graph with the test image centered at (0, 0). The coordinates of the model predictions have been adjusted to the relative center and the direction the agent is facing.  

# Remaining Goals and Challenges

## Goals 

We have demonstrated that performing coordinate regression prediction on a single image is a viable option. Then we would like to upscale our project by incorporating several images of a single coordinate. Our prototype only used images from 1 direction rather than 4 directions. Hopefully, the additional images will provide more information, which may decrease MSE. We have also shown in our status report that a simple CNN model outperforms all MLP models we have created. With the utilization of transfer learning and fine-tuning, we would like to investigate the performance of some CNN architectures, such as AlexNet, VGG, and LeNet-5 (the specific CNN architecture to be implemented is subjected to change), in our final report.

## Challenges

We anticipate 2 challenges: Machine Learning with multiple images of the same coordinate, and the time/computational cost of training large CNN architectures. We will address the problem of having to learn from multiple images of the same coordinate first. We will take 4 images (North, South, East, and West) at a 90 degree field of view in the same position and try to develop a machine learning model that performs multi-view learning. One potential issue is figuring out how we should adjust our models with the new data format. For example, the input shape for multi-view image data would be different from single image data (the one we did in this status report). We would solve this problem through reading some papers or looking at some models that implement some form of multi-view learning. The second challenge is the time and computational cost for training CNN architectures. We created a simple CNN for our status report, and it took around 4 hours to ran 5 fold cross-validation. We anticipating building larger networks with a larger dataset for our final report meaning the training time will get longer. One solution is to grayscale the image to reduce image dimensions. This problem is tricky because performing dimensionality reduction will cause the image to lose some features that may be relevant to regression. We would have to find the balance between feature preservation and dimensionality reduction. Another possible solution is to use pre-trained models by transfer learning to reduce the training time, which we do not build a network from scratch. 


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

[Keras, Regression, and CNNs](https://www.pyimagesearch.com/2019/01/28/keras-regression-and-cnns/)
