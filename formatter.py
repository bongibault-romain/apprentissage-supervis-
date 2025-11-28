
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def format_data(data: pandas.DataFrame, column_name: str, constraints: dict[int, (float, float)]):
    # Apply constraints
    for i in range(len(data)):
        value = data.at[i, column_name]

        for constraint, (min_val, max_val) in constraints.items():
            if value >= min_val and value < max_val:
                data.at[i, column_name] = constraint
        
    return data

def confusion(model, X, y_true, type):
    y_pred = model.predict(X)
    matrix = confusion_matrix(y_true, y_pred)

    print(f"Confusion Matrix ({type}):")
    print(matrix)

    # plot confusion matrix train
    plt.figure(figsize=(10, 7))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({type})")
    plt.colorbar()
    tick_marks = range(len(set(y_true)))
    plt.xticks(tick_marks, set(y_true), rotation=45)
    plt.yticks(tick_marks, set(y_true))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    # show values in matrix
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, matrix[i, j],
                    horizontalalignment="center",
                    color="white" if matrix[i, j] > thresh else "black")
    plt.savefig(f'confusion_{model}_matrix_{type}.png')
    plt.show()
