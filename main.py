#mosty copied from baseline.py lol
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import roc_auc_score
from ffl.ClassificationBenchmark import ClassificationBenchmark
from ffl import params
import numpy as np

class AddFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X, y=None):
        a = X[:,1] + X[:,2] + X[:,3] + X[:,4] + X[:,5]
        return np.c_[X, a]

num_pipeline = Pipeline([
    ('inputing', SimpleImputer(strategy='median')),
    # ('adding', AddFeature()),
    ('scaling', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('encoding', OneHotEncoder()),
    ('imputer', SimpleImputer(strategy='median'))
])

pipeline = ColumnTransformer([
    ('drop', 'drop', ['PassengerId', 'Cabin', 'Destination', 'Name']),
    ('num', num_pipeline, ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']),
    ('xat', cat_pipeline, ['CryoSleep', 'VIP', 'HomePlanet'])
])

def main():
    train = pd.read_csv('train.csv')
    train_y = train.Transported

    train.drop('Transported', axis=1)
    train = pipeline.fit_transform(train)

    # params = {
    #     'n_estimators': [10, 50, 100, 250, 500, 1000],
    #     'max_depth': [None, 10, 50, 100, 250],
    # }

    gs = GridSearchCV(SVC(), params.SVC, n_jobs=-1, scoring='f1')
    gs.fit(train, train_y)

#     rt = SVC(n_estimators=250, max_depth=10, n_jobs=-1)
# #
#     rt.fit(train, train_y)
    
    test = pd.read_csv('test.csv')
    pid = test.PassengerId

    test = pipeline.transform(test)
    pred = gs.predict(test)
    cc = pd.concat([pid, pd.DataFrame(pred)], ignore_index=True, axis=1)
    cc.to_csv('./final.csv', index=False)

if __name__ == "__main__":
    main()