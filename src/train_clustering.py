import pandas as pd
import numpy as np
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from feature_engineering import add_features

df = pd.read_csv("/home/abishek-murugan/GUVI/GUVI Projects/diamond-price-prediction/data/diamonds.csv")
df[['x','y','z']] = df[['x','y','z']].replace(0, np.nan)
df.dropna(inplace=True)

df = add_features(df)
cluster_df = df.drop(columns=['price'])
cluster_df = cluster_df.select_dtypes(include=np.number)

scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_df)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled)

joblib.dump(kmeans, "/home/abishek-murugan/GUVI/GUVI Projects/diamond-price-prediction/models/diamond_cluster_model.pkl")
joblib.dump(scaler, "/home/abishek-murugan/GUVI/GUVI Projects/diamond-price-prediction/models/scaler.pkl")
