# Customer Churn ML

Proyecto Integrador — Laboratorio de Minería de Datos (ISTEA, 2do cuatrimestre 2026)

Sistema de predicción de abandono de clientes (Customer Churn) para una empresa de
telecomunicaciones. Clasificación binaria (Yes/No) con scikit-learn, versionado de
datos con DVC, tracking de experimentos con MLflow, y despliegue como servicio REST
con FastAPI (etapas siguientes).

## Estado actual
Entrega 1 en desarrollo: EDA completo, pipeline de preprocesamiento y comparación de
6 modelos (baseline, lineal, árboles) registrados con MLflow.

## Reproducir el modelo inicial

Desde la carpeta principal del proyecto, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\dvc.exe pull
.\.venv\Scripts\python.exe -m src.training.train
```

`dvc pull` recupera el conjunto de datos histórico desde el remoto configurado; requiere acceso a ese remoto. El entrenamiento usa una división estratificada de 80 % para entrenar y 20 % para evaluar, con `random_state=42`.

Para ver las ejecuciones registradas localmente en MLflow:

```powershell
.\.venv\Scripts\mlflow.exe server --port 5000
```

Luego abrir http://localhost:5000 y seleccionar **Model training → customer-churn → Runs**. El modelo inicial obtuvo ROC-AUC de 0,812 y recall de 0,446 para la clase `Yes`.
