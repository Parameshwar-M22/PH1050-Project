#import libararies and modules
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyRegressor
def run_baseline(X, y, cv):
    model = DummyRegressor(strategy="mean")
    scores = cross_val_score(model,X,y.values.ravel(),cv=cv,scoring="neg_mean_absolute_error")
    mae_scores = -scores
    mean_mae = mae_scores.mean()

    return mean_mae