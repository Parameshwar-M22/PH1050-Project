# ---BASELINE MODEL -----
import numpy as np
#Define baseline prediction function
def baseline_predict(X_train,y_train, X_test):
    # Calculate the average boiling point from training data
    mean_value = np.mean(y_train)
    # Predict the same value for every test element
    predictions = np.full(len(X_test), mean_value)
    return predictions