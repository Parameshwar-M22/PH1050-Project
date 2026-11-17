#------------KNN ---------------
import numpy as np
def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    # We predict the boiling point of each test element one by one
    for test_point in X_test:
        # Calculate the distance between this element and all training elements
        distance = np.sqrt(
            np.sum((X_train - test_point)**2, axis=1)
        )
        # Get the indices of the k elements having the smallest distances (ie nearest)
        nearest_indices = np.argsort(distance)[:k] 
        # Find the boiling points of those nearest elements
        nearest_values = y_train[nearest_indices]
        # Take their average as the predicted boiling point
        prediction = np.mean(nearest_values)
        predictions.append(prediction)
    # Return all the predicted boiling points
    return np.array(predictions)
