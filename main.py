import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

features = pandas.read_csv("./alt_acsincome_ca_features_85.csv")
labels = pandas.read_csv("./alt_acsincome_ca_labels_85.csv")

X = np.array(features.values)
y = np.array(labels.get("PINCP").values)