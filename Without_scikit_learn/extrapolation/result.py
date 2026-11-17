import numpy as np


def evaluate_model(model, X_train, X_test, y_train, y_test,
                   scale=False, **kwargs):

    # Scale features if required
    if scale:
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)

        std[std == 0] = 1

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    # Train the model and predict the test data
    predictions = model(X_train, y_train, X_test, **kwargs)

    # Calculate Mean Absolute Error
    mae = np.mean(np.abs(y_test - predictions))

    return mae