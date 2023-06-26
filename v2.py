import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import TransformerMixin, BaseEstimator


class CabinFeatures(BaseEstimator, TransformerMixin):
    # USE IN COLUMN TRANSFORMER WITH CABIN AS THE ONLY INPUT
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        f1 = np.frompyfunc(lambda _: _.split('/')[0], 1, 1)
        f2 = np.frompyfunc(lambda _: _.split('/')[2], 1, 1)
        return np.c_[f1(X), f2(X)]


class PidFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        f = np.frompyfunc(lambda _: _.split('_')[1], 1, 1)
        return f(X)

class MakeDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]):
        self.cols = columns

    def fit(self, X, y=None):
        if X.shape[1] == len(self.cols):
            return self
        else:
            "no, of columns not equal to number of labels provided"
            raise IndexError
    def transform(self, X, y=None):
        return pd.DataFrame(X, columns=self.cols)


column_impute_pipeline = ColumnTransformer([
    ('mode_imputer', SimpleImputer(strategy='mode'), ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']),
    ('median_imputer', SimpleImputer(strategy='median'), ['RoomService', 'FoodCourt', 'ShoppingMall', 'VRDeck', 'Spa'])
])

column_prep_pipeline = ColumnTransformer([

])