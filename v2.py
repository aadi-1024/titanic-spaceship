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
        f = np.frompyfunc(lambda _: _.split('_')[0], 1, 1)
        f2 = np.frompyfunc(lambda _: np.count_nonzero(groups == _) == 1, 1, 1)
        groups = f(X)
        return f2(groups)
    

class MakeDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]):
        self.cols = columns

    def fit(self, X, y=None):
        if X.shape[1] == len(self.cols):
            return self
        else:
            print(X.shape[1], self.cols)
            "no, of columns not equal to number of labels provided"
            raise IndexError
    def transform(self, X, y=None):
        return pd.DataFrame(X, columns=self.cols)


column_impute_pipeline = ColumnTransformer([
    ('pass', 'passthrough', ['PassengerId']),
    ('mode_imputer', SimpleImputer(strategy='most_frequent'), ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']),
    ('median_imputer', SimpleImputer(strategy='median'), ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'VRDeck', 'Spa']),
    ('drop', 'drop', ['Name'])
])

column_prep_pipeline = ColumnTransformer([
    ('issolo', PidFeatures(), ['PassengerId']),
    ('cabin', CabinFeatures(), ['Cabin'])],
    remainder='passthrough'
)

scaling_encoding_pipeline = ColumnTransformer([
    ('scaling', StandardScaler(), ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'VRDeck', 'Spa']),
    ('encoding', OneHotEncoder(), ['is_solo', 'Cabin_deck', 'Cabin_side', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP'])
])

overall_pipeline = Pipeline([
    ('imputing', column_impute_pipeline),
    ('df', MakeDataFrame(['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'VRDeck', 'Spa'])),
    ('features', column_prep_pipeline),
    ('df2', MakeDataFrame(['is_solo', 'Cabin_deck', 'Cabin_side', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'VRDeck', 'Spa'])),
    ('scaling_encoding', scaling_encoding_pipeline)
])

def main():
    data = pd.read_csv('./train.csv')
    train_y = data.Transported
    train_X = overall_pipeline.fit_transform(data)

    params = {
        'n_estimators': [10, 50, 100, 250, 500, 1000],
        'max_depth': [None, 10, 50, 100, 250],
    }

    model = GridSearchCV(RandomForestClassifier(), params, n_jobs=10)
    model.fit(train_X, train_y)

    test = pd.read_csv('test.csv')
    pid = test.PassengerId

    test = overall_pipeline.transform(test)
    pred = model.predict(test)
    cc = pd.concat([pid, pd.DataFrame(pred)], ignore_index=True, axis=1)
    cc.to_csv('./final.csv', index=False)

if __name__ == '__main__':
    main()