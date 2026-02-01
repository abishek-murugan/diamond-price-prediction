import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['volume'] = df['x'] * df['y'] * df['z']
    df['price_per_carat'] = df['carat']  # placeholder at inference
    df['dimension_ratio'] = (df['x'] + df['y']) / (2 * df['z'])

    # Skewness handling (production-safe)
    df['log_carat'] = np.log1p(df['carat'])
    df['log_volume'] = np.log1p(df['volume'])

    return df