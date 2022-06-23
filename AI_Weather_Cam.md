---
layout: post
title: "AI Weather Cam"
description: "ONE LINER DESCRIPTION"
categories: envision
thumbnail: "filename.jpg"
year: 2022
---

### Project Guide

- (Add name and designation, if any)

### Mentors

- Vaishali S
- Nikhil P Reddy
- Harish Gumnur

### Members

- Fill This 
- Member2
- Member3

## Acknowledgements

(If you have a guide, acknowledge it here)

## Aim
Weather Image Classification has various important applications in our daily lives. With the current buzz being around Deep Learning, the use of Convolutional Neural Networks(CNNs) to classify these weather conditions comes as no surprise. To emphasise the diverse applicability of state-of-the-art CNNs, this project implements transfer learning of various architectures such as VGG16, ResNet and DenseNet and the results obtained from our classification using these models are compared at the end.

## Introduction

Severe weather conditions have a great impact on our daily lives. Automatic recognition of weather conditions has various important applications such as traffic condition warnings, automobile auxiliary driving, intelligent transportation systems, and other aspects. With the recent advancement of artificial intelligence, there might be some evidence that better weather forecasts can be produced by neural networks into the weather prediction workflow. With the rapid development of deep learning, deep convolutional neural networks (CNN) are used to recognize weather conditions. The convolutional layers are utilized to extract weather characteristics and finally, the weather images are classified and recognized through the fully connected layer and Softmax classifier. We use a dataset of weather images, which contains 6862 weather images spread across 11 categories: Dew, Fog/Smog, Frost, Glaze, Hail, Lightning, Rain, Rainbow, Rime, Sandstorm, and Snow. In this project, we trained our model based on various state-of-the-art architectures, such as ResNet, VGG16, DenseNet, and compared the subsequent results with useful visualization.

![image 1](/virtual-expo/assets/img/SIG/img1.jpg)

## Tranfer Learning 
Pre-trained models are frequently utilised as the foundation for deep learning tasks in computer vision and natural language processing because they save both time and money compared to developing neural network models from scratch and because they perform vastly better on related tasks.

A model created for one task is used as the basis for another using the machine learning technique known as transfer learning.

![Transfer Learning](/assets/transfer_learning.jpg)

## VGG-16
VGG 16 was proposed by Karen Simonyan and Andrew Zisserman of the Visual Geometry Group Lab of Oxford University in 2014 in the paper “VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION”. This model won 1st  and 2nd place in the above categories in the 2014 ILSVRC challenge.
This model achieves 92.7% top-5 test accuracy on the ImageNet dataset which contains 14 million images belonging to 1000 classes.

![Transfer Learning](/assets/VGG16.jpg)


## ResNet
In Deep CNN we come acrossa a very common problem of vanishing/exploding gradient. This causes the gradient to become 0 or too large. As we increases number of layers, the training and test error rate also increases. In order to solve the problem of the vanishing/exploding gradient, this architecture introduced the concept called Residual Blocks. In this network, we use a technique called skip connections. The skip connection connects activations of a  layer to further layers by skipping some layers in between. This forms a residual block. Resnets are made by stacking these residual blocks together. We used the Resnet 34 Model.

![Transfer Learning](/assets/ResNet.png)

## DenseNet
DenseNet is one of the new discoveries in neural networks for visual object recognition. DenseNet is quite similar to ResNet with some fundamental differences. ResNet uses an additive method (+) that merges the previous layer (identity) with the future layer, whereas DenseNet concatenates (.) the output of the previous layer with the future layer. We used the DenseNet169 model.

![Transfer Learning](/assets/densenet.png)


![Transfer Learning](/assets/densenet_info.png)

## Data Set
We used the [Weather Image Recognition Data Set](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) on Kaggle. This dataset contains 6862 images of 11 different classes of weather images.

- dew 
- fog/smog
- frost
- glaze
- hail
- lightning
- rain
- rainbow
- rime
- sandstorm
- snow

## Results


## Conclusion

- Conclusion1
- Conclusion2

## References

1. Some text, [Link](https://ieee.nitk.ac.in)
2. Some other text, [Link](https://ieee.nitk.ac.in)
3. Some more text, [Link](https://ieee.nitk.ac.in)
