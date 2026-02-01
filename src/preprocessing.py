from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

def build_preprocessor(categorical_cols, numerical_cols):
    return ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ]
    )
