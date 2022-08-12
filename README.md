# Bandit

Experimenting with simple convolutional neural networks for image classification. All of the functionality was implemented from scratch and Eigen was used for the linear algebra.

## Features

* Stochastic gradient descent
    * Vectorization
    * Momentum
    * L2 regularization
    * Weight initialization
* Layers: dense, convolutional
* Cost functions: quadratic, cross-entropy
* Activation functions: ReLu, leaky ReLu, sigmoid
* Data loaders: MNIST, CIFAR-100, ImageNet

## Performance

All the training-testing mini-batches were interleaved proportionally.

MNIST dataset:
* 97.7% testing accuracy in 2 minutes over 3 epochs
* 98.3% testing accuracy in 11 minutes over 15 epochs

## Images

![Screenshot](https://i.imgur.com/OeYNIii.png)
