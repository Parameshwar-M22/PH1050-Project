#import libararies and modules
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
def run_knn(X_train, y_train, X_test, y_test):
    model = make_pipeline(
        #Filling the missing EN values simply with the median of training data alone.
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsRegressor()
    )
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
