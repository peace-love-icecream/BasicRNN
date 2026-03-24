from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
import numpy as np
from Layers.Initializers import *


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # init hidden state with zeros
        self.hidden_state = []
        self.hidden_state.append(np.zeros((1, hidden_size)))
        self._memorize = False
        self._optimizer = None
        #self.tanH = TanH()
        #self.sigmoid = Sigmoid()
        #self.fullyconnected2 = FullyConnected(hidden_size, output_size)
        self._weights = np.ones((self.input_size + self.hidden_size + 1, self.hidden_size))
        self._weights2 = np.ones((hidden_size + 1, output_size))
        self.initialize(UniformRandom(), UniformRandom())
        self.output = None
        self.input = None
        self.fullyconnected1 = None
        self.fullyconnected2 = None
        self.tanH = None
        self.sigmoid = None
        self._gradient_weights = None
        self._gradient_weights2 = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_weights2(self):
        return self._gradient_weights2

    @property
    def weights(self):
        return self._weights

    @property
    def weights2(self):
        return self._weights2

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @gradient_weights2.setter
    def gradient_weights2(self, value):
        self._gradient_weights2 = value

    @weights.setter
    def weights(self, value):
        self._weights = value

    @weights2.setter
    def weights2(self, value):
        self._weights2 = value

    def forward(self, input_tensor):
        self.fullyconnected1 = [FullyConnected(self.input_size + self.hidden_size, self.hidden_size) for batch in input_tensor]
        self.tanH = [TanH() for batch in input_tensor]
        self.fullyconnected2 = [FullyConnected(self.hidden_size, self.output_size) for batch in input_tensor]
        self.sigmoid = [Sigmoid() for batch in input_tensor]

        hidden = self.hidden_state[len(self.hidden_state)-1]
        self.output = []

        for batch, fullyconnected1, tanH, fullyconnected2, sigmoid in zip(input_tensor, self.fullyconnected1,self.tanH, self.fullyconnected2, self.sigmoid):
            fullyconnected1.weights = self.weights
            fullyconnected2.weights = self.weights2
            #print("Weights", fullyconnected2.weights)
            input_concat = np.concatenate((batch, hidden[0]))
            input_concat = np.expand_dims(input_concat, axis=0)
            hidden = tanH.forward(fullyconnected1.forward(input_concat))
            if self.memorize:
                self.hidden_state.append(hidden)
            else:
                self.hidden_state.append(np.zeros((1, self.hidden_size)))
            yt = sigmoid.forward(fullyconnected2.forward(hidden))
            self.output.append(yt[0])

        self.output = np.array(self.output)
        return self.output

    def backward(self, error_tensor):
        backward_ht = np.zeros((1, self.hidden_size))
        self.gradient_weights = 0
        self.gradient_weights2 = 0
        output = []
        index = error_tensor.shape[0] - 1
        for i, batch in enumerate(reversed(error_tensor)):

            backward_sigmoid = self.sigmoid[index - i].backward(batch)

            backward_fully2 = self.fullyconnected2[index - i].backward(backward_sigmoid)

            backward_copy = backward_fully2 + backward_ht

            backward_tanh = self.tanH[index - i].backward(backward_copy)

            backward_fully1 = self.fullyconnected1[index - i].backward(backward_tanh) # gradient x_tilde


            # gradient_input
            backward_input = backward_fully1[0][0:self.input_size]
            backward_input = np.expand_dims(backward_input, axis=0)
            backward_ht = backward_fully1[0][self.input_size:]

            backward_ht = np.expand_dims(backward_ht, axis=0)

            # gradient_weights
            self.gradient_weights += self.fullyconnected1[index - i].gradient_weights
            self.gradient_weights2 += self.fullyconnected2[index - i].gradient_weights
            output.append(backward_input[0])

        output.reverse()
        output = np.array(output)

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.weights2 = self.optimizer.calculate_update(self.weights2, self.gradient_weights2)
        return output

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, self.input_size, self.output_size)
        # since bias is part of weights variable, bias.shape = weights[-1, :].shape
        self.weights[-1, :] = bias_initializer.initialize(self.weights[-1, :].shape, 1, self.output_size)

        self.weights2 = weights_initializer.initialize(self.weights2.shape, self.input_size, self.output_size)
        # since bias is part of weights variable, bias.shape = weights[-1, :].shape
        self.weights2[-1, :] = bias_initializer.initialize(self.weights2[-1, :].shape, 1, self.output_size)

