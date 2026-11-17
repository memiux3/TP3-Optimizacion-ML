"""Entrena el sistema final con TODOS los datos y lo exporta para el predictor del Grupo 5.

Se usa desde el notebook 05 y se ejecutó además en entornos con scikit-learn 1.2 a 1.9
(los pickles de sklearn no son portables entre versiones), generando un modelo por versión:

    python exportar_modelo.py data/gpu_resource_reservations.csv predictor_v1_Grupo_5_modelos

Sólo depende de numpy, pandas, scikit-learn y joblib (sin matplotlib), para poder correr
en entornos mínimos.
"""
import lzma
import os
import pickle
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predictor_v1_Grupo_5 as P  # noqa: E402

TARGET = "canceled_job"
UMBRAL = 0.49
# Hiperparámetros elegidos en el notebook 04 (E3, RandomizedSearchCV)
HP = dict(learning_rate=0.020640148115522096, max_iter=643, max_leaf_nodes=159, min_samples_leaf=12,
          l2_regularization=0.004582133690843357, max_features=0.5, class_weight="balanced", random_state=42)
NUM = ["hours_before_execution", "execution_year", "execution_month", "execution_week", "execution_day",
       "peak_gpu_hours", "offpeak_gpu_hours", "num_primary_processes", "num_auxiliary_processes",
       "num_background_processes", "returning_client", "previous_job_cancellations",
       "successful_previous_jobs", "resource_modifications", "queue_wait_time", "estimated_cost_per_hour",
       "reserved_storage_units", "special_resource_requests", "total_gpu_hours", "total_processes",
       "gpu_changed", "cost_total", "cost_per_process", "weekday", "dayofyear", "has_org", "has_agent",
       "prev_cancel_ratio"]
CAT = ["service_tier", "client_region", "client_type", "request_channel", "reserved_gpu_type",
       "assigned_gpu_type", "reservation_commitment", "workload_profile", "agent", "org"]


def crear_modelo():
    hp = dict(HP)
    if P._version_sklearn() < (1, 4):  # max_features en HGB existe desde sklearn 1.4
        hp.pop("max_features")
    if P._version_sklearn() >= (1, 3):
        enc = dict(unknown_value=-1, encoded_missing_value=-1, min_frequency=20)
    else:  # sklearn 1.2: sin min_frequency y HGB no acepta códigos negativos -> NaN (= faltante)
        enc = dict(unknown_value=np.nan, encoded_missing_value=np.nan)
    ct = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", **enc), CAT),
         ("num", "passthrough", NUM)],
        verbose_feature_names_out=False,
    )
    return make_pipeline(ct, HistGradientBoostingClassifier(categorical_features=list(range(len(CAT))), **hp))


def entrenar(X, y):
    """Modelo + memoria a partir de X (28+ columnas) e y (0/1)."""
    Xp = P._preparar(X)
    claves = P._clave(Xp)
    n_lote = pd.Series(claves).map(pd.Series(claves).value_counts()).to_numpy()
    modelo = crear_modelo()
    modelo.fit(P._agregar_features(Xp), np.asarray(y),
               histgradientboostingclassifier__sample_weight=1 / np.sqrt(n_lote))
    memoria = {k: float(v) for k, v in pd.Series(np.asarray(y, dtype=float)).groupby(claves).mean().items()}
    return modelo, memoria


def exportar(csv, carpeta):
    df = pd.read_csv(csv)
    modelo, memoria = entrenar(df, df[TARGET])
    os.makedirs(carpeta, exist_ok=True)
    ejemplo = P._preparar(df.head(5))
    proba = modelo.predict_proba(P._agregar_features(ejemplo))[:, 1]
    mayor, menor = P._version_sklearn()
    bundle = {"modelo": modelo, "umbral": UMBRAL,
              "ejemplo": {c: ejemplo[c].tolist() for c in ejemplo.columns},
              "proba_ejemplo": proba.tolist(),
              "info": {"grupo": 5, "filas": len(df), "sklearn": sklearn.__version__,
                       "numpy": np.__version__, "pandas": pd.__version__, "python": platform.python_version()}}
    ruta = os.path.join(carpeta, "modelo_sklearn_%d.%d.joblib" % (mayor, menor))
    joblib.dump(bundle, ruta, compress=("xz", 3), protocol=4)
    with lzma.open(os.path.join(carpeta, "memoria.pkl.xz"), "wb") as f:  # dict str -> float: portable
        pickle.dump(memoria, f, protocol=4)
    print("OK", ruta, "%.1f MB" % (os.path.getsize(ruta) / 1e6), "| memoria:", len(memoria), "reservas")
    return bundle


if __name__ == "__main__":
    exportar(sys.argv[1], sys.argv[2])
