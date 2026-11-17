#import libararies and modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
def run_linear_regression(X, y, cv):
    model = make_pipeline(
        #Filling the missing EN values simply with the median of training data alone.
        SimpleImputer(strategy="median"),  
        LinearRegression()
    )
    # Training  and evaluating the model using 5-fold cross-validation. The cv attribute is passed  in result.py while the function is being called.
    scores = cross_val_score(model, X, y.values.ravel(), cv=cv, scoring="neg_mean_absolute_error")
    #Converting negative to positive ones and finding mean
    mae_scores = -scores   
    mean_mae = mae_scores.mean()
    return mean_mae