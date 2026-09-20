import mlflow
import mlflow.sklearn

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

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        predictores,
        objetivo,
        test_size=0.2,
        random_state=42,
        stratify=objetivo,
    )

    X_ajuste, X_validacion, y_ajuste, y_validacion = train_test_split(
        X_entrenamiento,
        y_entrenamiento,
        test_size=0.25,
        random_state=42,
        stratify=y_entrenamiento,
    )

    candidatos = {
        "baseline_mayoritario": DummyClassifier(strategy="most_frequent"),
        "logistica_inicial": LogisticRegression(max_iter=1000),
        "bosque_inicial": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "logistica_regularizacion": LogisticRegression(
            C=0.1,
            max_iter=1000,
        ),
        "logistica_clases_balanceadas": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
        ),
        "bosque_profundidad_limitada": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
        ),
    }

    mlflow.set_experiment("customer-churn")

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
            y_validacion, predicciones, pos_label="Yes", zero_division=0
        )
        recall_yes = recall_score(
            y_validacion, predicciones, pos_label="Yes", zero_division=0
        )
        f1_yes = f1_score(
            y_validacion, predicciones, pos_label="Yes", zero_division=0
        )
        auc = roc_auc_score(y_validacion == "Yes", probabilidades_yes)

        matriz = confusion_matrix(
            y_validacion, predicciones, labels=["No", "Yes"]
        )
        verdaderos_no, falsas_alarmas = matriz[0]
        abandonos_no_detectados, abandonos_detectados = matriz[1]

        with mlflow.start_run(run_name=nombre) as run:
            mlflow.log_param("tipo_modelo", type(clasificador).__name__)
            mlflow.log_param("tamano_prueba", 0.2)
            mlflow.log_param("tamano_validacion_sobre_entrenamiento", 0.25)
            mlflow.log_param("semilla_particiones", 42)
            mlflow.log_param("filas_ajuste", len(X_ajuste))
            mlflow.log_param("filas_validacion", len(X_validacion))

            for parametro, valor in clasificador.get_params().items():
                mlflow.log_param(f"modelo_{parametro}", str(valor))

            mlflow.log_metric("validacion_accuracy", exactitud)
            mlflow.log_metric("validacion_precision_yes", precision_yes)
            mlflow.log_metric("validacion_recall_yes", recall_yes)
            mlflow.log_metric("validacion_f1_yes", f1_yes)
            mlflow.log_metric("validacion_roc_auc", auc)
            mlflow.log_metric(
                "validacion_abandonos_no_detectados",
                int(abandonos_no_detectados),
            )
            mlflow.log_metric(
                "validacion_abandonos_detectados",
                int(abandonos_detectados),
            )
            mlflow.log_metric(
                "validacion_falsas_alarmas",
                int(falsas_alarmas),
            )

            mlflow.log_text(
                f"Filas reales y columnas predichas: No, Yes\n"
                f"[[{verdaderos_no}, {falsas_alarmas}],\n"
                f" [{abandonos_no_detectados}, {abandonos_detectados}]]\n",
                "matriz_confusion_validacion.txt",
            )
            mlflow.sklearn.log_model(
                sk_model=modelo,
                name="modelo_churn",
                serialization_format="cloudpickle",
            )

            print(f"\n{nombre} | Run ID: {run.info.run_id}")
            print(
                f"Accuracy = {exactitud:.3f}; "
                f"Precision Yes = {precision_yes:.3f}; "
                f"Recall Yes = {recall_yes:.3f}; "
                f"F1 Yes = {f1_yes:.3f}; "
                f"ROC-AUC = {auc:.3f}"
            )
            print("Matriz de confusión (filas y columnas: No, Yes):")
            print(matriz)

    print(f"\nClientes de validación: {len(X_validacion)}")
    print(f"Clientes reservados para prueba: {len(X_prueba)}")


if __name__ == "__main__":
    main()