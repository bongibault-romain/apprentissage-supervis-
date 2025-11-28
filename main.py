import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from formatter import confusion
import time

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

X = np.array(df_final.values)
y = np.array(labels.get("PINCP").values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

print("Starting Grid Search for RandomForestClassifier...")
# Dataset size
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Avoid nested or excessive parallelism which can exhaust memory/CPU
# Set n_jobs=1 here and for GridSearchCV to prevent spawning too many processes
model = GradientBoostingClassifier(random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [10, 20],
    'learning_rate': [1.0],
    'max_depth': [5, 7],
    'min_samples_split': [10, 15],
    'n_iter_no_change': [2, 4]
}

# Calculate total number of fits for progress bar
n_combinations = (
    len(param_grid['n_estimators']) *
    len(param_grid['learning_rate']) *
    len(param_grid['max_depth']) *
    len(param_grid['min_samples_split']) *
    len(param_grid['n_iter_no_change'])
)
cv_folds = 5
total_fits = n_combinations * cv_folds
print(f"Total combinations: {n_combinations}, Total fits (with CV): {total_fits}")

# Progress bar callback
class TqdmCallback:
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="GridSearch progress")
        self.n_completed = 0
    
    def __call__(self, result):
        self.pbar.update(1)
    
    def close(self):
        self.pbar.close()

progress_callback = TqdmCallback(total_fits)

# Grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,                 # 5-fold cross-validation
    n_jobs=-1,             # run grid search single-threaded to avoid OOM on small/limited machines
    pre_dispatch='2*n_jobs',
    scoring='accuracy',   # metric to optimize
    verbose=3             # verbose=3 shows progress for each fit
)

# Compute CPU time

start_time = time.process_time()

try:
    grid_search.fit(X_train, y_train)
except MemoryError:
    print("MemoryError: the grid search likely exhausted available RAM.\n"
          "Suggestions: set GridSearchCV(n_jobs=1), reduce the size of `param_grid`, use `RandomizedSearchCV`,\n"
          "or run on a machine with more memory. Stopping to avoid crashing the system.")
    raise

end_time = time.process_time()

print(f"Grid Search completed in {end_time - start_time} seconds.")
# Best hyperparameters
print("Best Parameters:", grid_search.best_params_)

# Best model
best_model = grid_search.best_estimator_

# Test accuracy
accuracy = best_model.score(X_test, y_test)
train_accuracy = best_model.score(X_train, y_train)

print("Test Accuracy:", accuracy)
print("Train Accuracy:", train_accuracy)
    
confusion(best_model, X_test, y_test, "Test")
confusion(best_model, X_train, y_train, "Train")