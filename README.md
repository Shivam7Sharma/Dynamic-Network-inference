
# README

This repository contains a PyTorch implementation of a branchy neural network for image classification on the CIFAR-10 dataset. The network is designed to have multiple exit points, allowing for early termination of inference based on a specified cutoff value.

## Network Architecture

The network consists of a series of convolutional and pooling layers, followed by a series of fully connected layers. Each convolutional and pooling layer is followed by a branch, which is a smaller neural network that predicts the output class probabilities. The branches are used to determine whether to exit the network early or continue to the next layer.

## Training and Evaluation

The network is trained on the CIFAR-10 training set and evaluated on the CIFAR-10 test set. The performance of the network is measured in terms of accuracy, total time, and total time vs overall accuracy.

## Cutoff Exit Performance Check

The `cutoff_exit_performance_check` function is used to evaluate the performance of the network for a given cutoff value. The function takes in the model, dataloader, cutoff value, and device as inputs and returns the accuracy and total time for each layer.

## Estimating Thresholds

The `estimate_thresholds` function is used to estimate the thresholds for each layer based on a desired minimum accuracy. The function takes in the model, dataloader, desired accuracy, and device as inputs and returns the estimated thresholds and layer-wise inference times.

## Testing with Thresholds

The `test_with_thresholds` function is used to test the network with the estimated thresholds. The function takes in the model, dataloader, thresholds, and device as inputs and returns the accuracy and total time.

## Plotting

The repository includes several plotting functions to visualize the performance of the network. The plots include overall accuracy vs cutoff, total time vs cutoff, total time vs overall accuracy, and inference time vs desired accuracy.

## TODO

1. (a) For a fixed value of cutoff, show performance for all layers.
2. (b) Plot overall accuracy vs cutoff, total time vs cutoff, and total time vs overall accuracy.
3. (c) Vary the desired minimum accuracy and generate lists of thresholds. For the list of list of thresholds, plot total time vs overall accuracy.

## Requirements

* PyTorch
* NumPy
* Matplotlib
* Scikit-learn
* SciPy

## Usage

1. Clone the repository and install the required dependencies.
2. Run the `Dynamic Network inference.py` script to train the network.
3. Run the `Dynamic Network inference.py` script to evaluate the network.
4. Run the `Dynamic Network inference.py` script to visualize the performance of the network.
