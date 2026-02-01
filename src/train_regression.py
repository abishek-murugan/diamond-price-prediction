import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from preprocessing import build_preprocessor
from feature_engineering import add_features

df = pd.read_csv("/home/abishek-murugan/GUVI/GUVI Projects/diamond-price-prediction/data/diamonds.csv")
df[['x','y','z']] = df[['x','y','z']].replace(0, np.nan)
df.dropna(inplace=True)

df['price_inr'] = df['price'] * 83
df = add_features(df)

X = df.drop(columns=['price', 'price_inr'])
y = df['price_inr']

categorical_cols = ['cut','color','clarity']
numerical_cols = X.drop(columns=categorical_cols).columns

preprocessor = build_preprocessor(categorical_cols, numerical_cols)

models = {
    "lr": LinearRegression(),
    "dt": DecisionTreeRegressor(max_depth=10),
    "rf": RandomForestRegressor(n_estimators=200, random_state=42),
    "knn": KNeighborsRegressor(n_neighbors=7)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipelines = {}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    pipelines[name] = pipe

joblib.dump(pipelines["rf"], "/home/abishek-murugan/GUVI/GUVI Projects/diamond-price-prediction/models/diamond_price_model.pkl")

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

X_ann = preprocessor.fit_transform(X)
y_ann = y.values

model = Sequential([
    Dense(128, activation="relu", input_shape=(X_ann.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

model.fit(
    X_ann, y_ann,
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=5)],
    verbose=1
)

model.save("/home/abishek-murugan/GUVI/GUVI Projects/diamond-price-prediction/models/diamond_price_ann_model.h5")