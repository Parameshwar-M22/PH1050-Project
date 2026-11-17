#import libraries and modules.
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
def run_baseline(X_train, y_train, X_test, y_test):
    model = DummyRegressor(strategy="mean")
    #Model training
    model.fit(X_train, y_train.values.ravel())
    #Model testing
    predictions = model.predict(X_test)
    #Finding the MAE from existing dataset.
    mae = mean_absolute_error(
        y_test.values.ravel(),
        predictions
    )
    return mae