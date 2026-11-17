# TP3 — Optimización de Modelos de ML (Grupo 2)

Predicción de cancelación de reservas de GPU (`canceled_job`). Sólo scikit-learn.

## Notebooks (correr en orden, desde esta carpeta)

| Notebook | Consigna | Contenido |
|---|---|---|
| `01_EDA_Preprocesamiento.ipynb` | 0 y 1 | leakage, missing, outliers, duplicados, EDA, cohortes, split 80/20 por grupos |
| `02_Baseline_CV.ipynb` | 2.1–2.3 | métricas (Balanced Accuracy + ROC-AUC), CV por grupos, 6 clasificadores, grid search, baseline |
| `03_Analisis_Errores.ipynb` | 3.1–3.4 | errores por cohorte, FP/FN, curvas de aprendizaje, cohorte prioritaria, adquisición de datos |
| `04_Optimizacion.ipynb` | 4 | 6 estrategias (FE, ponderación, HPO, votación, umbral, memoria) vs. baseline |
| `05_Modelo_Final.ipynb` | 5 | evaluación en test, mejora por cohorte, rendimiento esperado, export del predictor |

`tp3_comun.py`: definiciones compartidas (columnas, features, preprocesador, CV, métricas).

## Entrega del predictor

Enviar **juntos** `predictor_v1_Grupo_2.py` y `predictor_v1_Grupo_2.joblib` (misma carpeta).

```python
from predictor_v1_Grupo_2 import predictor_v1_Grupo_2
y_pred = predictor_v1_Grupo_2(X)   # X: DataFrame con las columnas del dataset -> array de 0/1
```

- Ignora columnas extra (incluidas `canceled_job`, `execution_status`, `execution_status_date`).
- Requiere scikit-learn 1.9.1 (versión con la que se serializó el modelo).

## Resultados

| | Balanced Accuracy | ROC-AUC |
|---|---|---|
| Baseline (HGB, 28 variables), test | 0.856 | 0.938 |
| Sistema final, test (reservas nuevas) | **0.858** | **0.940** |
| Sistema final, CV aleatoria (misma fuente) | **0.886** | 0.962 |
