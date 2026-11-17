"""Predictor del Grupo 2 — TP3 Aprendizaje Automático (ITBA 72.75).

Predice si una reserva de GPU será cancelada (1) o no (0).

Uso:
    from predictor_v1_Grupo_2 import predictor_v1_Grupo_2
    y_pred = predictor_v1_Grupo_2(X)      # X: DataFrame con las columnas del dataset

Requisitos: el archivo `predictor_v1_Grupo_2.joblib` en la misma carpeta que este .py,
y las librerías numpy, pandas, scikit-learn (ver requirements.txt).

El archivo es autocontenido: no depende del notebook ni de otros módulos del proyecto.
El modelo ya está entrenado; esta función NO reentrena ni usa etiquetas.

Sistema (ver notebooks 04 y 05):
  1. Limpieza: se usan sólo las 28 variables del enunciado (se ignoran `canceled_job`,
     `execution_status`, `execution_status_date` y cualquier otra columna).
  2. Features nuevas calculadas fila a fila (`_agregar_features`).
  3. HistGradientBoostingClassifier (preprocesamiento incluido en el Pipeline de sklearn).
  4. Memoria de reservas vistas: si la reserva es idéntica (en las 28 variables) a una del
     conjunto de entrenamiento, se usa la tasa de cancelación observada para esa reserva.
  5. Umbral de decisión optimizado para balanced accuracy.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_RUTA_MODELO = Path(__file__).resolve().parent / "predictor_v1_Grupo_2.joblib"
_BUNDLE = None

FEATURES_ORIGINALES = [
    "hours_before_execution", "execution_year", "execution_month", "execution_week",
    "execution_day", "peak_gpu_hours", "offpeak_gpu_hours", "num_primary_processes",
    "num_auxiliary_processes", "num_background_processes", "service_tier", "client_region",
    "client_type", "request_channel", "returning_client", "previous_job_cancellations",
    "successful_previous_jobs", "reserved_gpu_type", "assigned_gpu_type",
    "resource_modifications", "reservation_commitment", "orchestration_agent_id",
    "organization_id", "queue_wait_time", "workload_profile", "estimated_cost_per_hour",
    "reserved_storage_units", "special_resource_requests",
]
CATEGORICAS = [
    "service_tier", "client_region", "client_type", "request_channel",
    "reserved_gpu_type", "assigned_gpu_type", "reservation_commitment", "workload_profile",
]
NUMERICAS = [c for c in FEATURES_ORIGINALES if c not in CATEGORICAS]


def _preparar(X):
    """Deja sólo las 28 variables con tipos normalizados (numéricas -> float, categóricas -> str)."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=FEATURES_ORIGINALES)
    faltan = [c for c in FEATURES_ORIGINALES if c not in X.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en X: {faltan}")
    X = X[FEATURES_ORIGINALES].copy().reset_index(drop=True)
    for c in NUMERICAS:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    for c in CATEGORICAS:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        X[c] = X[c].map(lambda v: v if pd.isna(v) else str(v))
    return X


def _firma(Xp):
    """Huella de cada reserva (las 28 variables) para la memoria de reservas vistas."""
    clave = Xp.copy()
    for c in CATEGORICAS:
        clave[c] = clave[c].fillna("<NA>").astype(str)
    return pd.util.hash_pandas_object(clave, index=False).to_numpy()


def _agregar_features(X):
    """Mismas reglas fila a fila que tp3_comun.agregar_features (copiadas para ser autocontenido)."""
    X = X.copy()
    X["total_gpu_hours"] = X["peak_gpu_hours"] + X["offpeak_gpu_hours"]
    X["total_processes"] = X[["num_primary_processes", "num_auxiliary_processes",
                              "num_background_processes"]].sum(axis=1)
    X["gpu_changed"] = (X["reserved_gpu_type"] != X["assigned_gpu_type"]).astype(int)
    X["cost_total"] = X["estimated_cost_per_hour"] * X["total_gpu_hours"]
    X["cost_per_process"] = X["estimated_cost_per_hour"] / X["total_processes"].replace(0, np.nan)
    fecha = pd.to_datetime(
        dict(year=X["execution_year"], month=X["execution_month"], day=X["execution_day"]),
        errors="coerce",
    )
    X["weekday"] = fecha.dt.weekday
    X["dayofyear"] = fecha.dt.dayofyear
    X["has_org"] = X["organization_id"].notna().astype(int)
    X["has_agent"] = X["orchestration_agent_id"].notna().astype(int)
    total_prev = X["previous_job_cancellations"] + X["successful_previous_jobs"]
    X["prev_cancel_ratio"] = X["previous_job_cancellations"] / total_prev.replace(0, np.nan)
    X["agent"] = X["orchestration_agent_id"].fillna(-1).astype(int).astype(str)
    X["org"] = X["organization_id"].fillna(-1).astype(int).astype(str)
    return X


def _cargar():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(_RUTA_MODELO)
    return _BUNDLE


def predecir_proba_Grupo_2(X):
    """Probabilidad de cancelación (modelo + memoria de reservas vistas)."""
    b = _cargar()
    Xp = _preparar(X)
    proba = b["modelo"].predict_proba(_agregar_features(Xp))[:, 1]
    memoria = pd.Series(_firma(Xp)).map(b["memoria"]).to_numpy(dtype=float)
    return np.where(np.isnan(memoria), proba, memoria)


def predictor_v1_Grupo_2(X):
    """Recibe X (DataFrame con las columnas del dataset; columnas extra se ignoran)
    y devuelve un np.ndarray de 0/1 (1 = la reserva será cancelada)."""
    return (predecir_proba_Grupo_2(X) >= _cargar()["umbral"]).astype(int)
