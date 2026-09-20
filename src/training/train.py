import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.load_data import load_historical_data, split_features_target
from src.features.preprocess import build_preprocessor


def main():
    data = load_historical_data()
    X, y = split_features_target(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probability_yes = model.predict_proba(X_test)[
        :, list(model.classes_).index("Yes")
    ]

    auc = roc_auc_score(y_test == "Yes", probability_yes)
    recall_yes = recall_score(y_test, predictions, pos_label="Yes")

    print(classification_report(y_test, predictions, zero_division=0))
    print(f"ROC-AUC: {auc:.3f}")
    print(f"Clientes usados para entrenar: {len(X_train)}")
    print(f"Clientes reservados para evaluar: {len(X_test)}")

    mlflow.set_experiment("customer-churn")
    with mlflow.start_run(run_name="logistic-regression-baseline"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("recall_yes", recall_yes)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="churn_model",
            serialization_format="cloudpickle",
        )


if __name__ == "__main__":
    main()