#import libraries and modules
from sklearn.model_selection import KFold
#creating the 5 fold cross validation mechanism
cv = KFold(n_splits=5, shuffle=True, random_state=42)
#importing the functions and data set from other py files
from data_model import X, y
from linear_regression import run_linear_regression
from baseline import run_baseline
from knn import run_knn
from decision_tree import run_decision_tree
#Calling the functions that implement the alogirthm and passing the dataset and cv into the called functions.
baseline_mae = run_baseline(X, y, cv)
lr_mae = run_linear_regression(X, y, cv)
knn_mae = run_knn(X, y, cv,5)
dst_mae = run_decision_tree(X, y, cv)
print("-------------CROSS VALIDATION RESULT-------------")
print("Linear Regression MAE:", lr_mae)
print("Baseline MAE:", baseline_mae)
print("KNN MAE:", knn_mae)
print("DST MAE:", dst_mae)