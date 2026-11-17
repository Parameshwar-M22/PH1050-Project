# ----------LINEAR REGRESSION------------
import numpy as np
def linear_predict(X_train, y_train, X_test):
    # Start with all weights and bias as zero
    weights = np.zeros(X_train.shape[1])
    bias = 0
    learning_rate = 0.01
    epochs = 10000
    n = len(X_train)

    # Keep updating the weights to reduce the prediction error
    for i in range(epochs):
        # Calculate predictions using current weights, perform matrix multiplication 
        predictions = X_train @ weights + bias

        # Find the difference between actual and predicted values
        error = predictions - y_train

        # Calculate how much each weight and bias should change
        dw = (2/n) * (X_train.T @ error)
        db = (2/n) * np.sum(error)

        # Update weights and bias to reduce MSE
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

    # Use the final weights to predict boiling points of test elements
    predictions = X_test @ weights + bias

    return predictions
