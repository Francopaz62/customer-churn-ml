from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_historical_data():
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "raw" / "customer_churn_historical.csv"
    data = pd.read_csv(csv_path)
    return data

def split_features_target(data):
    X = data.drop(columns=["customerID", "Churn"])
    y = data["Churn"]
    return X, y

if __name__ == "__main__":
    data = load_historical_data()
    X, y = split_features_target(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Entrenamiento: {X_train.shape}")
    print(f"Prueba: {X_test.shape}")
    print(f"Predictores: {X.shape}")
    print(f"Objetivo: {y.shape}")
    print(f"Filas: {len(data)}")
    print(f"Columnas: {len(data.columns)}")
    print("\nValores faltantes por columna:")
    print(data.isna().sum())
    print(f"\nTipo de TotalCharges: {data['TotalCharges'].dtype}")
    print(data.loc[data["TotalCharges"].isna(), ["tenure", "MonthlyCharges", "TotalCharges"]].head(10))
    print("Faltantes con tenure distinto de 0:")
    print(data.loc[data["TotalCharges"].isna() & (data["tenure"] != 0)].shape[0])
    print("\nTipos de los predictores:")
    print(X.dtypes)