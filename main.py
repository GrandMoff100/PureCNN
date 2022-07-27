import pandas as pd
from cnn.net import ConvolutionNeuralNetwork


df = pd.read_csv("digits.csv", comment="#")
df["label"] = df["label"].astype("category")
