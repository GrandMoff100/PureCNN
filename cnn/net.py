import itertools
from collections import defaultdict

import numpy as np

from cnn.func import convolution, pool, relu, relu_prime


class Layer:
    """Abstract class for network layers."""

    def execute(self, inp: np.ndarray) -> np.ndarray:
        """Abstract method for evaluating a layer."""
        raise NotImplementedError()


class MaxPool(Layer):
    """Implements a max pooling layer."""

    def __init__(self, size: int = 2):
        self.size = size

    def execute(self, inp: np.ndarray) -> np.ndarray:
        return pool(inp, (self.size,) * 2)


class FlattenLayer(Layer):
    """Implements a flattening layer."""

    def execute(self, inp: np.ndarray) -> np.ndarray:
        return inp.resize((inp.size,))


class LearningLayer(Layer):
    """Abstract class for backpropagation layers."""

    def backpropagate(
        self, inp: np.ndarray, out: np.ndarray, truth: np.ndarray
    ) -> np.ndarray:
        """Abstract method for backpropogating a layer."""
        raise NotImplementedError()

    def step(self, batch_size: int, **kwargs) -> None:
        """Abstract method for updating the weights and biases of a layer."""
        raise NotImplementedError()


class ConvolutionalLayer(LearningLayer):
    """Implements a Kernel Convolution layer with a bias and activation."""

    def __init__(self, previous_width: int, width: int):
        self.previous_width = previous_width
        self.width = width
        self.kernel = np.zeros((self.previous_width - self.width,) * 2)
        self.bias = 0

    def execute(self, inp: np.ndarray) -> np.ndarray:
        return relu(convolution(inp, self.kernel) + self.bias)

    def backpropagate(
        self, inp: np.ndarray, out: np.ndarray, truth: np.ndarray
    ) -> np.ndarray:
        """Abstract method for backpropogating a layer."""
        raise NotImplementedError()


class FCLayer(LearningLayer):
    def __init__(self, size: int, next_size: int, activation=relu):
        self.size = size
        self.next_size = next_size
        self.activation = activation
        self.weights = np.random.rand(next_size, size)
        self.biases = np.random.rand(next_size)

    def execute(self, inp: np.ndarray):
        return self.activation(self.weights @ inp + self.biases)

    def backpropagate(
        self, inp: np.ndarray, out: np.ndarray, truth: np.ndarray
    ) -> np.ndarray:
        return (
            self.weights_gradient(inp, out, truth),
            self.bias_gradient(inp, out, truth),
            self.previous_layer_gradient(inp, out, truth),
        )

    def previous_layer_gradient(
        self, inp: np.ndarray, out: np.ndarray, truth: np.ndarray
    ) -> np.ndarray:
        return self.weights.sum(axis=1) * relu_prime(self.weights @ inp + self.biases) * (out - truth)

    def bias_gradient(self, inp: np.ndarray, out: np.ndarray, truth: np.ndarray) -> np.ndarray:
        return relu_prime(self.weights @ inp + self.biases) * (out - truth)

    def weights_gradient(self, inp: np.ndarray, out: np.ndarray, truth: np.ndarray) -> np.ndarray:
        return relu_prime(self.weights @ inp + self.biases) @ (out - truth) * self.weights

    def step(
        self,
        weights: np.ndarray,
        biases: np.ndarray,
        batch_size: int,
    ) -> None:
        """Update the weights and biases of the network."""
        self.weights -= weights / batch_size
        self.biases -= biases / batch_size


class Network:
    """Implements a Neural Network."""

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def execute(self, training_set: np.ndarray) -> np.ndarray:
        """Execute the network."""
        previous_layer = training_set
        for layer in self.layers:
            previous_layer = layer.execute(previous_layer)
        return previous_layer

    def loss(self, out: np.ndarray, label: np.ndarray) -> float:
        """Calculate the loss."""
        return np.sum(np.abs(out - label) ** 2)

    def batch_loss(self, pixel_data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate the loss for a batch."""
        losses = np.array([])
        for image, label in zip(pixel_data, labels):
            losses = np.append(losses, self.loss(self.execute(image), label))
        return losses.mean()

    def accumulate_layer_activations(self, image: np.ndarray) -> np.ndarray:
        """Accumulate the activations of the network."""
        previous_layers = [image]
        for layer in self.layers:
            previous_layers.append(layer.execute(previous_layers[-1]))
        return previous_layers[1:]

    def train_batch(self, images: np.ndarray, labels: np.ndarray):
        """Train the network with a single batch."""
        gradients = defaultdict(lambda: defaultdict(int))
        for image, layer_truth in zip(images, labels):
            activations = self.accumulate_layer_activations(image)
            for layer, (activation, previous_activation) in zip(
                self.layers[::-1], itertools.pairwise(activations[::-1] + [image])
            ):
                if isinstance(layer, FCLayer):
                    weights_gradient, bias_gradient, layer_truth = layer.backpropagate(
                        previous_activation, activation, layer_truth
                    )
                    gradients[layer]["weights"] += weights_gradient
                    gradients[layer]["biases"] += bias_gradient
                elif isinstance(layer, ConvolutionalLayer):
                    raise NotImplementedError(
                        "Convolutional layers are not implemented yet."
                    )
        for layer, gradients in gradients.items():
            layer.step(**gradients, batch_size=labels.size)

    def train(
        self,
        pixel_data: np.ndarray,
        labels: np.ndarray,
        iterations: int = 1000,
        batch_size: int = 10,
    ):
        """Train the network with minibatches."""
        for i in range(iterations):
            for batch, batch_labels in zip(
                np.array_split(pixel_data, batch_size),
                np.array_split(labels, batch_size),
            ):
                self.train_batch(batch, batch_labels)
                print("Iteration", i, self.batch_loss(batch, batch_labels))
