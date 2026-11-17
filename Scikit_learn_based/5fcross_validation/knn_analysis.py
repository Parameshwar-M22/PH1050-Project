#import libraries and modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
#creating the 5 fold cross validation mechanism
cv = KFold(n_splits=5, shuffle=True, random_state=42)
#importing the functions and data set from other py files
from data_model import X, y
from linear_regression import run_linear_regression
from baseline import run_baseline
from knn import run_knn
from decision_tree import run_decision_tree
knn_mae = np.zeros(20)
n_values = np.arange(1, 21)

for i in range(20):
    n = n_values[i]
    knn_mae[i] = run_knn(X, y, cv, n)

plt.plot(n_values, knn_mae, marker='o')
plt.xlabel("Number of Neighbours (n)")
plt.ylabel("Cross-Validation MAE")
plt.title("KNN: Neighbours vs CV MAE")
plt.grid(True)
plt.show()