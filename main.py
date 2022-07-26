import numpy as np
import pandas as pd

from cnn.net import ConvolutionalLayer, FCLayer, FlattenLayer, MaxPool, Network

df = pd.read_csv("digits.csv", comment="#")


def one_hot(label: str) -> np.ndarray:
    """Convert a label to a one-hot vector."""
    return np.array([1 if i == label else 0 for i in "0123456789"])


df["label"] = df["label"].apply(one_hot)

data = df.drop(columns=["label"]).to_numpy()
labels = df["label"].to_numpy()


network = Network(
    [
        FCLayer(784, 10),
    ]
)


network.train(data, labels, iterations=1000, batch_size=10)
