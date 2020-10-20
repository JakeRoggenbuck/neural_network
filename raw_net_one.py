import numpy as np


def signmoid(x):
    return 1.0/(1 + np.exp(-x))


def signmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = signmoid(np.dot(self.input, self.weights1))
        self.output = signmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * signmoid_derivative(self.output), self.weights2.T) * signmoid_derivative(self.layer1))
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * signmoid_derivative(self.output)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])

    nn = NeuralNetwork(X,y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()

    print(nn.output)
