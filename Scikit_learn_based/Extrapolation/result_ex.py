#import dataset(the splitted one), and the alogrithms
from .data_split import X_train, y_train, X_test, y_test
from .baseline_ex import run_baseline
from .linear_regression_ex import run_linear_regression
from .knn_ex import run_knn
from .decision_tree_ex import run_decision_tree
#Calling the functions that implement the alogirthm and passing the dataset and cv into the called functions.
baseline_mae = run_baseline(X_train, y_train, X_test, y_test)
lr_mae = run_linear_regression(X_train, y_train, X_test, y_test)
knn_mae = run_knn(X_train, y_train, X_test, y_test)
dst_mae = run_decision_tree(X_train, y_train, X_test, y_test)
print("-----------EXTRAPOLATION RESULTS------------")
print("Baseline MAE:", baseline_mae)
print("Linear Regression MAE:", lr_mae)
print("KNN MAE:", knn_mae)
print("Decision Tree MAE:", dst_mae)

