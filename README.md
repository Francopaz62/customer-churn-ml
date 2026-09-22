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

El acceso al proyecto en DagsHub requiere ser colaborador invitado: no alcanza con que el repositorio esté marcado como público, hay que pedir acceso al owner en https://dagshub.com/Francopaz62/customer-churn-ml.

Una vez con acceso, configurar las credenciales localmente (no se suben a Git, viven en `.dvc/config.local`, que está en `.gitignore`):

```powershell
.\.venv\Scripts\dvc.exe remote modify origin --local auth basic
.\.venv\Scripts\dvc.exe remote modify origin --local user <tu_usuario_de_DagsHub>
.\.venv\Scripts\dvc.exe remote modify origin --local password <tu_token_de_DagsHub>
```

El token se genera en DagsHub: perfil → Settings → Tokens → Generate New Token.

Desde la carpeta principal del proyecto, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\dvc.exe pull
.\.venv\Scripts\python.exe -m src.training.train
```

`dvc pull` recupera el conjunto de datos histórico desde el remoto configurado; requiere acceso a ese remoto. El entrenamiento usa una división estratificada de 80 % para entrenar y 20 % para evaluar, con `random_state=42`.

Las ejecuciones se registran en DagsHub (no localmente). Para verlas, abrir:

https://dagshub.com/Francopaz62/customer-churn-ml/experiments

El modelo inicial obtuvo ROC-AUC de 0,812 y recall de 0,446 para la clase `Yes`.

## Comparación y selección del modelo

Se separó un 20 % de los datos como conjunto de prueba. Con el 80 % restante se hizo otra división para ajustar los modelos y compararlos sobre un conjunto de validación de 1409 clientes. Las divisiones fueron estratificadas y se utilizó `random_state=42`. La comparación se ejecuta con:

```powershell
.\.venv\Scripts\python.exe -m src.training.comparacion_modelos
```

| Modelo | Precisión Yes | Recall Yes | F1 Yes | ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline mayoritario | 0,000 | 0,000 | 0,000 | 0,500 |
| Regresión logística inicial | 0,651 | 0,477 | 0,551 | 0,822 |
| Bosque aleatorio inicial | 0,618 | 0,396 | 0,483 | 0,794 |
| Regresión logística con regularización | 0,650 | 0,466 | 0,543 | 0,823 |
| Regresión logística con clases balanceadas | 0,495 | 0,774 | 0,604 | 0,821 |
| Bosque aleatorio con profundidad limitada | 0,670 | 0,377 | 0,483 | 0,810 |

Se eligió como candidato la **regresión logística con clases balanceadas**, registrada en MLflow como `modelo_abandono_clientes`, versión 1. Su ejecución de origen es `logistica_clases_balanceadas` (Run ID `01744985dee54171858142d2627bd893`, commit `19d588e`).

En validación detectó 287 de los 371 clientes que abandonaron y dejó 84 sin detectar. La regresión logística inicial detectó 177 y dejó 194 sin detectar. Elegir el modelo balanceado permite identificar más posibles abandonos, pero también aumenta las falsas alarmas de 95 a 293. La elección supone que contactar a un cliente que finalmente permanece tiene un costo menor que no detectar a uno que abandona. Esa prioridad de negocio debe confirmarse antes de usar el modelo en una campaña real.

Estas métricas corresponden a **validación**; no son resultados del conjunto de prueba final.