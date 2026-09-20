from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.load_data import load_historical_data, split_features_target
from src.features.preprocess import build_preprocessor


def main():
    datos = load_historical_data()
    predictores, objetivo = split_features_target(datos)

    # Se reserva el 20 % para la evaluación final.
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        predictores,
        objetivo,
        test_size=0.2,
        random_state=42,
        stratify=objetivo,
    )

    # La validación permite comparar modelos sin usar la prueba final.
    X_ajuste, X_validacion, y_ajuste, y_validacion = train_test_split(
        X_entrenamiento,
        y_entrenamiento,
        test_size=0.25,
        random_state=42,
        stratify=y_entrenamiento,
    )

    candidatos = {
        "Baseline mayoritario": DummyClassifier(strategy="most_frequent"),
        "Regresión logística": LogisticRegression(max_iter=1000),
        "Bosque aleatorio": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    for nombre, clasificador in candidatos.items():
        modelo = Pipeline(
            [
                ("preprocessor", build_preprocessor(X_ajuste)),
                ("classifier", clasificador),
            ]
        )

        modelo.fit(X_ajuste, y_ajuste)

        predicciones = modelo.predict(X_validacion)
        indice_yes = list(modelo.classes_).index("Yes")
        probabilidades_yes = modelo.predict_proba(X_validacion)[:, indice_yes]

        exactitud = accuracy_score(y_validacion, predicciones)
        precision_yes = precision_score(
            y_validacion,
            predicciones,
            pos_label="Yes",
            zero_division=0,
        )
        recall_yes = recall_score(
            y_validacion,
            predicciones,
            pos_label="Yes",
            zero_division=0,
        )
        f1_yes = f1_score(
            y_validacion,
            predicciones,
            pos_label="Yes",
            zero_division=0,
        )
        auc = roc_auc_score(
            y_validacion == "Yes",
            probabilidades_yes,
        )

        matriz = confusion_matrix(
            y_validacion,
            predicciones,
            labels=["No", "Yes"],
        )

        print(f"\n{nombre}")
        print(
            f"Accuracy = {exactitud:.3f}; "
            f"Precision Yes = {precision_yes:.3f}; "
            f"Recall Yes = {recall_yes:.3f}; "
            f"F1 Yes = {f1_yes:.3f}; "
            f"ROC-AUC = {auc:.3f}"
        )
        print("Matriz de confusión (filas reales, columnas predichas):")
        print("Orden de clases: No, Yes")
        print(matriz)

    print(f"\nClientes de validación: {len(X_validacion)}")
    print(f"Clientes reservados para la prueba final: {len(X_prueba)}")


if __name__ == "__main__":
    main()