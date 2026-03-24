from Layers.Base import BaseLayer
import numpy as np
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable = True
        self._optimizer = None
        self.bias = None
        self.weights = None
        self.mean_b = None
        self.sigma_b = None
        self.input = None
        self._moving_mean = None
        self._moving_sigma = None
        # decay rate
        self.alpha = 0.8
        self._gradient_weights = 0
        self._gradient_bias = 0
        self.normalized_input = None
        self.initialize(None, None)
        self.B = None
        self.H = None
        self.N = None
        self.M = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def optimizer(self):
        """optimizer property"""
        return self._optimizer

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def moving_mean(self):
        return self._moving_mean

    @moving_mean.setter
    def moving_mean(self, value):
        self._moving_mean = value

    @property
    def moving_sigma(self):
        return self._moving_sigma

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @moving_sigma.setter
    def moving_sigma(self, value):
        self._moving_sigma = value

    def forward(self, input_tensor):
        if len(input_tensor.shape) == 4:
            input = self.reformat(input_tensor)
        else:
            input = input_tensor
        # in test case only calculate normalized input and output
        if self.testing_phase:
            normalized_input_tensor = (input - self.moving_mean) / (np.sqrt(self.moving_sigma + np.finfo(float).eps))
            if len(input_tensor.shape) == 4:
                return self.reformat(self.weights * normalized_input_tensor + self.bias)
            return self.weights * normalized_input_tensor + self.bias

        # save input tensor
        self.input = input

        # save and calculate mean + variance for batch
        self.mean_b = np.mean(input, axis=0)
        self.sigma_b = np.var(input, axis=0)

        # initialize moving mean + variance
        if self.moving_mean is None:
            self.moving_mean = self.mean_b
            self.moving_sigma = self.sigma_b

        # save and calculate moving mean + variance
        self.moving_mean = self.alpha * self.moving_mean + (1 - self.alpha) * self.mean_b
        self.moving_sigma = self.alpha * self.moving_sigma + (1 - self.alpha) * self.sigma_b

        # save and calculate normalized input tensor
        self.normalized_input = (input - self.mean_b) / (np.sqrt(self.sigma_b + np.finfo(float).eps))

        if len(input_tensor.shape) == 4:
            return self.reformat(np.multiply(self.weights, self.normalized_input) + self.bias)

        return np.multiply(self.weights, self.normalized_input) + self.bias

    def backward(self, error_tensor):
        if len(error_tensor.shape) == 4:
            error = self.reformat(error_tensor)
        else:
            error = error_tensor

        # gradient with respect to W
        self.gradient_weights = np.sum(error * self.normalized_input, axis=0)

        # update weights
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        # gradient with respect to bias
        self.gradient_bias = np.sum(error, axis=0)

        # gradient with respect to X
        if len(error_tensor.shape) == 4:
            return self.reformat(compute_bn_gradients(error, self.input, self.weights, self.mean_b, self.sigma_b))

        return compute_bn_gradients(error, self.input, self.weights, self.mean_b, self.sigma_b)

    def initialize(self, _, __):
        self.bias = np.zeros(self.channels)
        self.weights = np.ones(self.channels)

    def reformat(self, tensor):
        # convert from 4 to 2 dimensions
        if len(tensor.shape) == 4:
            # save input shape for recovery
            self.B, self.H, self.N, self.M = tensor.shape
            # flatten spatial dimension
            tensor1 = np.moveaxis(tensor.reshape(self.B, self.H, self.N * self.M), -1, 1)
            tensor2 = tensor1.reshape(self.B * self.N * self.M, self.H)
            return tensor2
        # convert from 2 to 4 dimensions
        else:
            tensor1 = tensor.reshape(self.B, self.N * self.M, self.H)
            tensor2 = np.moveaxis(tensor1, 1, -1).reshape((self.B, self.H, self.N, self.M))
            return tensor2

