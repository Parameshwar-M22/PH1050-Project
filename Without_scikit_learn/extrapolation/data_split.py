import numpy as np
import sys
from pathlib import Path

cv_path = Path(__file__).resolve().parent.parent / "5fold_cross validation"
sys.path.append(str(cv_path))

from data_model import df, X, y

def split_data():

    # Use elements from Periods 1 to 4 for training
    train_mask = df["Period"].to_numpy() <= 4

    # Use elements from Periods 5 and 6 for testing
    test_mask = df["Period"].to_numpy() > 4

    # Convert features and target into NumPy arrays
    features = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    target = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    target = target.ravel()

    X_train = features[train_mask]
    X_test = features[test_mask]

    y_train = target[train_mask]
    y_test = target[test_mask]

    return X_train, X_test, y_train, y_test