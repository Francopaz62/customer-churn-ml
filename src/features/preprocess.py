from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X):
    numeric_columns = X.select_dtypes(include="number").columns
    categorical_columns = X.select_dtypes(exclude="number").columns

    return ColumnTransformer([
        ("numeric", StandardScaler(), numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ])