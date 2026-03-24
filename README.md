# Deep Learning Framework from Scratch

A lightweight, layer-oriented deep learning framework built from scratch using NumPy. This project builds upon BasicCNN by adding regularization techniques, batch normalization, dropout, and recurrent neural network components.

## Overview

This framework extends the convolutional neural network implementation (BasicCNN) with essential regularization strategies and sequential modeling capabilities. It follows the same layer-oriented architecture where each layer implements `forward()` and `backward()` operations. New additions include:

- **Regularization**: L1 and L2 weight constraints with configurable regularization strength
- **Dropout**: Inverted dropout for preventing co-adaptation of neurons
- **Batch Normalization**: Trainable normalization with moving average for test time
- **Recurrent Layers**: Elman RNN cells for sequence processing
- **Activation Functions**: TanH and Sigmoid for recurrent architectures
- **Persistence**: Pickle-based save/load functionality for trained networks
