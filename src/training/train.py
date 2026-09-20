from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, roc_auc_score

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
    print(classification_report(y_test, predictions, zero_division=0))
    probability_yes = model.predict_proba(X_test)[:, list(model.classes_).index("Yes")]
    print(f"ROC-AUC: {roc_auc_score(y_test == 'Yes', probability_yes):.3f}")
    print(f"Clientes usados para entrenar: {len(X_train)}")
    print(f"Clientes reservados para evaluar: {len(X_test)}")


if __name__ == "__main__":
    main()