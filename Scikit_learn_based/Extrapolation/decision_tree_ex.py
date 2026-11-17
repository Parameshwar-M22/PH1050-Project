#import libararies and modules
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
def run_decision_tree(X_train, y_train, X_test, y_test):
    model = make_pipeline(
        #Filling the missing EN values simply with the median of training data alone.      
        SimpleImputer(strategy="median"),
        DecisionTreeRegressor(random_state=42)
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
