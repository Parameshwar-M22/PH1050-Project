#import libararies and modules
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
def run_knn(X, y, cv,n):

    model = make_pipeline(
        #Filling the missing EN values simply with the median of training data alone.
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=n)
    )

    scores = cross_val_score(model,X,y.values.ravel(),cv=cv,scoring="neg_mean_absolute_error")
    mae_scores = -scores
    mean_mae = mae_scores.mean()

    return mean_mae
