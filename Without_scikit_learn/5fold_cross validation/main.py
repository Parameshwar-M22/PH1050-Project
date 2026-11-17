#----MASTER FILE TO DISPLAY RESULTS------
from data_model import X, y
from result import evaluate_model
from knn import knn_predict
from baseline import baseline_predict
from linear_regressor import linear_predict

# Evaluate baseline
baseline_mae = evaluate_model(baseline_predict, X, y)

# Evaluate KNN with k = 3
knn_mae = evaluate_model(knn_predict, X, y, k=3)
#Linear regression model
linear_mae = evaluate_model(linear_predict, X, y)


print("------------THE MAE RESULTS-----------")
print("Baseline MAE:", baseline_mae)
print("KNN MAE:", knn_mae)
print("Linear Regression MAE:", linear_mae)
