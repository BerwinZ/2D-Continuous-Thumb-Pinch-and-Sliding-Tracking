"""
This script is to train the KNN model on the main components
It could runs in the raspbeery pi
"""
# ---------------------------------------------
# 1. Initialization
# ---------------------------------------------

components_path = r"./models/knn/components.csv"
model_folder = r"./models/knn"

print("Import Packages...")

from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib 

# ---------------------------------------------
# 2. Prepare data
# ---------------------------------------------
print("\nReading Data...")

components_pd = pd.read_csv(components_path)
X = components_pd.iloc[:, 0].to_numpy().reshape(-1, 1)
Y = components_pd.iloc[:, 1: 3].to_numpy()

print("Original dataset shape")
print("X:", X.shape, " Y:", Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
print("Train dataset shape")
print("X:", X_train.shape, " Y:", Y_train.shape)
print("Test dataset shape")
print("X:", X_test.shape, " Y:", Y_test.shape)

# ---------------------------------------------
# 3. Prepare model
# ---------------------------------------------
neighbor_num = 30
neigh_x = KNeighborsRegressor(n_neighbors=neighbor_num)
neigh_y = KNeighborsRegressor(n_neighbors=neighbor_num)

# ---------------------------------------------
# 4. Fit and test models
# ---------------------------------------------

print("\nStart training...")
neigh_x.fit(X_train, Y_train[:, 0])
neigh_y.fit(X_train, Y_train[:, 1])

print("------------------------------------")
print("Scores")
sc1 = neigh_x.score(X_test, Y_test[:, 0])
sc2 = neigh_y.score(X_test, Y_test[:, 1])

print(sc1, sc2)

# ---------------------------------------------
# 5. Save models
# ---------------------------------------------
model_path1 = model_folder + r'/' + 'knn_x.joblib'
model_path2 = model_folder + r'/' + 'knn_y.joblib'
print("Saving in", model_path1)
print("Saving in", model_path2)

joblib.dump(neigh_x, model_path1)
joblib.dump(neigh_y, model_path2)

