# TP3 — Optimización de Modelos de ML (Grupo 5)

Predicción de cancelación de reservas de GPU (`canceled_job`). Sólo scikit-learn.

## Notebooks (correr en orden, desde esta carpeta)

| Notebook | Consigna | Contenido |
|---|---|---|
| `01_EDA_Preprocesamiento.ipynb` | 0 y 1 | leakage, missing, outliers, duplicados, EDA, cohortes, split 80/20 por grupos |
| `02_Baseline_CV.ipynb` | 2.1–2.3 | métricas (Balanced Accuracy + ROC-AUC), CV por grupos, 6 clasificadores, grid search, baseline |
| `03_Analisis_Errores.ipynb` | 3.1–3.4 | errores por cohorte, FP/FN, curvas de aprendizaje, cohorte prioritaria, adquisición de datos |
| `04_Optimizacion.ipynb` | 4 | 6 estrategias (FE, ponderación, HPO, votación, umbral, memoria) vs. baseline |
| `05_Modelo_Final.ipynb` | 5 | evaluación en test, estadísticas finales, mejora por cohorte, rendimiento esperado, predictor |

- `tp3_comun.py`: definiciones compartidas (columnas, features, preprocesador, CV, métricas).
- `exportar_modelo.py`: entrena el sistema final con todos los datos y exporta el modelo.

## Entrega del predictor

Enviar **juntos** (misma carpeta) — o el ZIP `entrega_predictor_Grupo_5.zip`:

```
predictor_v1_Grupo_5.py
predictor_v1_Grupo_5_modelos/
```

```python
from predictor_v1_Grupo_5 import predictor_v1_Grupo_5
y_pred = predictor_v1_Grupo_5(X)   # X: DataFrame con las columnas del dataset -> array de 0/1
```

- Ignora columnas extra (incluidas `canceled_job`, `execution_status`, `execution_status_date`).
- Funciona con **scikit-learn 1.2 a 1.9** (numpy 1.x o 2.x, pandas 1.5 a 3.x): hay un modelo exportado
  por versión y el predictor carga el que corresponde, verificándolo con reservas de control.
  Probado en 11 entornos (Python 3.10 a 3.14, incluido uno equivalente a Google Colab).

## Resultados

| | Train | Validación (CV) | **Test** |
|---|---|---|---|
| Balanced Accuracy | 0.959 | 0.856 | **0.858** |
| ROC-AUC | 0.993 | 0.942 | **0.940** |

Baseline (HGB, 28 variables) en test: BA 0.856, AUC 0.938. Estimación realista para la competición
(reservas nuevas y únicas, en línea con el leaderboard del 11/11): BA ≈ 0.80–0.83.
