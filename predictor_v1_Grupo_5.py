"""Predictor del Grupo 5 — TP3 Aprendizaje Automático (ITBA 72.75).

Predice si una reserva de GPU será cancelada (1) o no (0).

Uso:
    from predictor_v1_Grupo_5 import predictor_v1_Grupo_5
    y_pred = predictor_v1_Grupo_5(X)      # X: DataFrame con las columnas del dataset

Archivos necesarios (en la misma carpeta que este .py):
    predictor_v1_Grupo_5.py
    predictor_v1_Grupo_5_modelos/   (modelos entrenados + memoria)

Requisitos: numpy, pandas, scikit-learn >= 1.2 (cualquier versión 1.2 a 1.9), joblib.
El modelo se guardó con cada versión de scikit-learn de 1.2 a 1.9; el predictor carga
automáticamente el que corresponde a la versión instalada (los pickles de scikit-learn
no son compatibles entre versiones). No reentrena ni usa etiquetas.

Sistema (ver notebooks 04 y 05):
  1. Sólo se usan las 28 variables del enunciado (se ignoran `canceled_job`,
     `execution_status`, `execution_status_date` y cualquier otra columna).
  2. Features nuevas calculadas fila a fila (`_agregar_features`).
  3. HistGradientBoostingClassifier (preprocesamiento dentro del Pipeline de sklearn).
  4. Memoria de reservas vistas: si la reserva es idéntica (28 variables) a una del
     dataset de entrenamiento, se usa la tasa de cancelación observada.
  5. Umbral de decisión optimizado para balanced accuracy.
"""
import lzma
import os
import pickle
import warnings

import numpy as np
import pandas as pd

_DIR_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictor_v1_Grupo_5_modelos")
_CACHE = {}

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
        raise ValueError("Faltan columnas en X: %s" % faltan)
    X = X[FEATURES_ORIGINALES].copy().reset_index(drop=True)
    for c in NUMERICAS:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    for c in CATEGORICAS:
        X[c] = pd.Series([v if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
                          for v in X[c].astype(object)], dtype=object)
        X[c] = X[c].where(X[c].notna(), np.nan)
    return X


def _clave(Xp):
    """Clave de texto de cada reserva (28 variables). Independiente de versiones de pandas/numpy."""
    partes = []
    for c in FEATURES_ORIGINALES:
        if c in CATEGORICAS:
            partes.append(["<NA>" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
                           for v in Xp[c].tolist()])
        else:
            partes.append(["nan" if np.isnan(v) else repr(float(v)) for v in Xp[c].tolist()])
    return ["|".join(fila) for fila in zip(*partes)]


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
        pd.DataFrame({"year": X["execution_year"], "month": X["execution_month"], "day": X["execution_day"]}),
        errors="coerce",
    )
    X["weekday"] = fecha.dt.weekday
    X["dayofyear"] = fecha.dt.dayofyear
    X["has_org"] = X["organization_id"].notna().astype(int)
    X["has_agent"] = X["orchestration_agent_id"].notna().astype(int)
    total_prev = X["previous_job_cancellations"] + X["successful_previous_jobs"]
    X["prev_cancel_ratio"] = X["previous_job_cancellations"] / total_prev.replace(0, np.nan)
    X["agent"] = X["orchestration_agent_id"].fillna(-1).astype(int).astype(str).astype(object)
    X["org"] = X["organization_id"].fillna(-1).astype(int).astype(str).astype(object)
    return X


def _version_sklearn():
    import sklearn
    partes = sklearn.__version__.split(".")
    return int(partes[0]), int("".join(ch for ch in partes[1] if ch.isdigit()))


def _candidatos():
    """Archivos de modelo ordenados: primero el de la versión instalada, luego los más cercanos."""
    actual = _version_sklearn()
    lista = []
    for nombre in os.listdir(_DIR_MODELOS):
        if nombre.startswith("modelo_sklearn_") and nombre.endswith(".joblib"):
            mayor, menor = nombre[len("modelo_sklearn_"):-len(".joblib")].split(".")
            v = (int(mayor), int(menor))
            lista.append((v != actual, abs((v[0] - actual[0]) * 100 + v[1] - actual[1]), v < actual, nombre))
    return [n for *_, n in sorted(lista)]


def _cargar():
    if "modelo" in _CACHE:
        return _CACHE
    import joblib
    with lzma.open(os.path.join(_DIR_MODELOS, "memoria.pkl.xz"), "rb") as f:
        _CACHE["memoria"] = pickle.load(f)
    errores = []
    for nombre in _candidatos():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                b = joblib.load(os.path.join(_DIR_MODELOS, nombre))
                # prueba: el modelo cargado debe reproducir la probabilidad guardada del ejemplo
                p = b["modelo"].predict_proba(_agregar_features(_preparar(pd.DataFrame(b["ejemplo"]))))[:, 1]
            if not np.allclose(p, b["proba_ejemplo"], atol=1e-6):
                raise ValueError("predicción de control distinta")
            _CACHE.update(modelo=b["modelo"], umbral=b["umbral"], archivo=nombre)
            return _CACHE
        except Exception as e:  # versión incompatible -> probar la siguiente
            errores.append("%s: %s" % (nombre, repr(e)[:200]))
    raise RuntimeError("No se pudo cargar ningún modelo compatible:\n" + "\n".join(errores))


def predecir_proba_Grupo_5(X):
    """Probabilidad de cancelación (modelo + memoria de reservas vistas)."""
    b = _cargar()
    Xp = _preparar(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = b["modelo"].predict_proba(_agregar_features(Xp))[:, 1]
    memoria = np.array([b["memoria"].get(k, np.nan) for k in _clave(Xp)], dtype=float)
    return np.where(np.isnan(memoria), proba, memoria)


def predictor_v1_Grupo_5(X):
    """Recibe X (DataFrame con las columnas del dataset; columnas extra se ignoran)
    y devuelve un np.ndarray de 0/1 (1 = la reserva será cancelada)."""
    return (predecir_proba_Grupo_5(X) >= _cargar()["umbral"]).astype(int)
