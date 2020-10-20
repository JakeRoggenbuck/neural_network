import numpy as np


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.rand(self.input.shape[1], 4)
        self.weights2 = np.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)