import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

num_pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('inputing', SimpleImputer(strategy='median'))
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

    rt = RandomForestClassifier()
    train = pipeline.fit_transform(train)

    rt.fit(train, train_y)

    test = pd.read_csv('test.csv')
    pid = test.PassengerId

    test = pipeline.transform(test)
    pred = rt.predict(test)
    cc = pd.concat([pid, pd.DataFrame(pred)], ignore_index=True, axis=1)
    cc.to_csv('./final.csv', index=False)

if __name__ == "__main__":
    main()