import sys
from pathlib import Path
cv_path = Path(__file__).resolve().parent.parent / "5fold_cross validation"
sys.path.append(str(cv_path))
from data_split import split_data
from result import evaluate_model
from knn import knn_predict
from baseline import baseline_predict
from linear_regressor import linear_predict


# Get the training and testing datasets
X_train, X_test, y_train, y_test = split_data()


# Evaluate the baseline model
baseline_mae = evaluate_model(
    baseline_predict,
    X_train, X_test, y_train, y_test
)


# Evaluate KNN
knn_mae = evaluate_model(
    knn_predict,
    X_train, X_test, y_train, y_test,
    scale=True,
    k=3
)


# Evaluate Linear Regression
linear_mae = evaluate_model(
    linear_predict,
    X_train, X_test, y_train, y_test,
    scale=True
)




# Display the results
print("-----Extrapolation Results-----------")
print("Baseline MAE:", baseline_mae)
print("KNN MAE:", knn_mae)
print("Linear Regression MAE:", linear_mae)
