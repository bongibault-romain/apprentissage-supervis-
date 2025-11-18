import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

features = pandas.read_csv("./dataset/alt_acsincome_ca_features_85.csv")
labels = pandas.read_csv("./dataset/alt_acsincome_ca_labels_85.csv")

X = np.array(features.values)
y = np.array(labels.get("PINCP").values)

def distribution(name, column):
    fig, ax = plt.subplots()

    # foreach value in column, count
    values = {}

    for v in column:
        if not v in values:
            values[v] = 0
        
        values[v] += 1

    # fruits = ['apple', 'blueberry', 'cherry', 'orange']
    # counts = [40, 100, 30, 55]

    ax.bar(values.keys(), values.values())

    ax.set_ylabel('value')
    ax.set_title(name)

    plt.show()
    plt.close(fig)
    
distribution("AGEP", features.get("AGEP").values)
distribution("COW", features.get("COW").values) # 1 - Autre
distribution("SCHL", features.get("SCHL").values)
distribution("MAR", features.get("MAR").values)
distribution("OCCP", features.get("OCCP").values)
distribution("POBP", features.get("POBP").values)
distribution("RELP", features.get("RELP").values)
distribution("WKHP", features.get("WKHP").values)
distribution("SEX", features.get("SEX").values)
distribution("RAC1P", features.get("RAC1P").values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)