#-----CALCULATE MAE VALUES-----------
import numpy as np
from cross_validation import five_fold
def evaluate_model(model, X, y,scale="True", **kwargs):

    mae_scores = []
    # Run the model on all five folds
    for X_train, X_test, y_train, y_test in five_fold(X, y):
        if scale:
            # Standardize features using training data so that when some models like KNN are used, large values don't cause large distance
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)

            #Handle the edge case where a feature has constant values
            std[std == 0] = 1

            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std

        # Get predictions from the selected model
        predictions = model(X_train, y_train, X_test, **kwargs)

        # Calculate MAE for the current fold
        mae = np.mean(np.abs(y_test - predictions))

        mae_scores.append(mae)

    # Calculate the average MAE across all five folds
    return np.mean(mae_scores)

