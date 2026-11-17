#---------CREATING 5 FOLDS AND CREATING THE TRAINING AND TESTING DATASETS --------
import numpy as np
def five_fold(X, y):
    indices = np.arange(len(X))
    #Shuffling so that we can remove biases
    np.random.seed(42)
    np.random.shuffle(indices)
    #Create 5 equal folds to implement 5Fold method.
    folds = np.array_split(indices, 5)
    for i in range(5):
        test_indices = folds[i]   #choosing a particular index for testing
        train_indices = np.concatenate([folds[j] for j in range(5) if j != i])  #keeping the other 4 for training

        X_train = X[train_indices]
        X_test = X[test_indices]

        y_train = y[train_indices]
        y_test = y[test_indices]

        yield X_train, X_test, y_train, y_test
