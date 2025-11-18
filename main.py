import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

features = pd.read_csv("./dataset/alt_acsincome_ca_features_85.csv")
labels = pd.read_csv("./dataset/alt_acsincome_ca_labels_85.csv")

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

cols_to_encode = ['MAR', 'SEX', 'COW', 'OCCP', 'RAC1P', 'POBP', 'SCHL', 'RELP']

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(features[cols_to_encode])

encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cols_to_encode))

df_final = pd.concat([features.drop(columns=cols_to_encode), encoded_df], axis=1)

print(features.head())
print(df_final.head())
#plt_x_sex = features.get("SEX").values
#plt_y_pinpc = labels.get("AGEP").values

# plot correlation
#plt.scatter(plt_x_sex, plt_y_pinpc)
#plt.xlabel("SEX")
#plt.ylabel("AGEP")
#plt.title("Correlation between SEX and AGEP")
#plt.show()


# format_data(features, "AGEP", {
#     0: (0, 20),
#     1: (20, 40),
#     2: (40, 60),
#     3: (60, 120),
# })

# distribution("AGEP", features.get("AGEP").values)
# distribution("COW", features.get("COW").values) # 1 - Autre
# distribution("SCHL", features.get("SCHL").values)
# distribution("MAR", features.get("MAR").values)
# distribution("OCCP", features.get("OCCP").values)
# distribution("POBP", features.get("POBP").values)
# distribution("RELP", features.get("RELP").values)
# distribution("WKHP", features.get("WKHP").values)
# distribution("SEX", features.get("SEX").values)
# distribution("RAC1P", features.get("RAC1P").values)

X = np.array(df_final.values)
y = np.array(labels.get("PINCP").values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)