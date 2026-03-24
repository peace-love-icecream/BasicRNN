import copy
from scipy.signal import correlate
from scipy.signal import convolve
from Layers.Base import BaseLayer
import numpy as np


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.random(np.insert(convolution_shape, 0, num_kernels))
        self.bias = np.random.rand(num_kernels)
        self._optimizer = None
        self._bias_optimizer = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, val):
        self._bias_optimizer = copy.deepcopy(val)
        self._optimizer = val

    def forward(self, input_tensor):
        self.input_tensor = copy.deepcopy(input_tensor)
        batch_result = []
        input_tensor = self.pad_xy(input_tensor, self.convolution_shape)
        for b in range(input_tensor.shape[0]):
            result = None
            for x in range(self.num_kernels):
                image = input_tensor[b]
                convolved = correlate(image, self.weights[x], mode="valid")
                convolved = convolved + self.bias[x]

                if (len(self.convolution_shape) > 2):
                    convolved = convolved[:, ::self.stride_shape[0], ::self.stride_shape[1]]
                else:
                    convolved = convolved[:, ::self.stride_shape[0]]

                if (result is None):
                    result = convolved
                else:
                    result = np.concatenate((result, convolved), axis=0)

            batch_result.append(result)
        return np.array(batch_result)

    def backward(self, error_tensor):
        # upsampling
        result_shape = np.array(self.input_tensor.shape)
        result_shape[1] = self.num_kernels
        padded = np.zeros(result_shape, dtype=error_tensor.dtype)

        if(len(self.convolution_shape)>2):
            padded[:,:,::self.stride_shape[0],::self.stride_shape[1]] = error_tensor
        else:
            padded[:, :, ::self.stride_shape[0]] = error_tensor
        error_tensor = padded
        #compute gradient_weights
        input_tensor = self.pad_xy(self.input_tensor, self.convolution_shape)
        self.gradient_weights = []
        for b in range(error_tensor.shape[0]):
            one_element = []
            for n in range(self.num_kernels):
                error_tensor_plane = np.expand_dims(error_tensor[b][n],axis=0)
                correlation = correlate(input_tensor[b], error_tensor_plane, mode="valid")
                one_element.append(correlation)
            self.gradient_weights.append(one_element)
        self.gradient_weights = np.sum(np.array(self.gradient_weights),axis=0)

        if (self.optimizer != None):
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        # gradient_bias: sum over axis: batch, width, height
        if(len(self.convolution_shape)>2):
            self.gradient_bias = error_tensor.sum(axis=(0, 2, 3))
        else:
            self.gradient_bias = error_tensor.sum(axis=(0,2))
        if (self.optimizer != None):
            self.bias = self._bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        # compute gradient with respect to the lower layers:
        # compute new kernels (plane of the output contributes to the corresponding layer in all the kernels)
        all_new_kernels = []
        for x in range(self.convolution_shape[0]):
            new_kernel = []
            for y in range(self.num_kernels):
                new_kernel.append(self.weights[y][x])
            new_kernel = np.flip(np.array(new_kernel), axis=0)
            all_new_kernels.append(new_kernel)
        # pad
        error_tensor = self.pad_xy(error_tensor, all_new_kernels[0].shape)
        # compute derivative with new kernels for every element in batch
        batch = []
        for b in range(error_tensor.shape[0]):
            gradient_lower_layers = None
            for k in range(len(all_new_kernels)):
                convolved = convolve(error_tensor[b], all_new_kernels[k], mode="valid")
                if gradient_lower_layers is None:
                    gradient_lower_layers = convolved
                else:
                    gradient_lower_layers = np.concatenate((gradient_lower_layers, convolved), axis=0)
            batch.append(gradient_lower_layers)
        return np.array(batch)

    def pad_xy(self, input_tensor, convolution_shape):
        # +1 for batch dimension
        npad = np.zeros((len(convolution_shape) + 1, 2))
        for x in range(1, len(convolution_shape)):
            # x+1 because of batch dimension
            npad[x + 1] = (convolution_shape[x] / 2, np.ceil(convolution_shape[x] / 2) - 1)
        input_tensor = np.pad(input_tensor, np.int_(npad))
        return input_tensor

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.convolution_shape[0] * self.convolution_shape[1] * self.convolution_shape[2]
        fan_out = self.num_kernels * self.convolution_shape[1] * self.convolution_shape[2]
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)


