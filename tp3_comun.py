"""Definiciones compartidas entre los notebooks del TP3 (GPU reservations, `canceled_job`).

Igual que en el TP2:
- Las reglas FIJAS (columnas a eliminar, features nuevas calculadas fila a fila) no
  aprenden nada de los datos -> se pueden aplicar a train y test sin leakage.
- Las transformaciones que SÍ aprenden (codificación, escalado, imputación) se crean con
  `crear_preprocesador()` y se ajustan siempre DENTRO de un Pipeline, es decir, dentro
  de cada fold de la validación cruzada.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "canceled_job"
DATA_DIR = Path("data")
FIG = Path("figuras")
RESULTADOS = Path("resultados")
CSV_ORIGINAL = DATA_DIR / "gpu_resource_reservations.csv"

# ---------------------------------------------------------------- columnas
# Información POSTERIOR a la reserva: dicen directamente si se canceló -> leakage.
LEAKAGE = ["execution_status", "execution_status_date"]
# Columnas que no figuran en el diccionario del enunciado. total_* son sumas exactas
# de otras columnas (las recalculamos nosotros); high_value_client no está documentada.
NO_DOCUMENTADAS = ["total_processes", "total_gpu_hours", "high_value_client"]

# Las 28 variables del diccionario del enunciado (páginas 4-5).
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
CAT_ORIGINALES = [
    "service_tier", "client_region", "client_type", "request_channel",
    "reserved_gpu_type", "assigned_gpu_type", "reservation_commitment", "workload_profile",
]
NUM_ORIGINALES = [c for c in FEATURES_ORIGINALES if c not in CAT_ORIGINALES]

# Cohortes que pide analizar la consigna (3.1)
COHORTES = ["client_type", "service_tier", "reserved_gpu_type"]


# ---------------------------------------------------------------- datos
def cargar_original():
    """Lee el CSV completo (con todas las columnas, incluidas las de leakage)."""
    return pd.read_csv(CSV_ORIGINAL)


def limpiar(df):
    """Reglas fijas de limpieza: quita leakage y columnas no documentadas, deja las 28."""
    return df[FEATURES_ORIGINALES].copy()


def cargar_split(nombre):
    """Lee la partición guardada en el notebook 01 ('train' o 'test') -> (X, y)."""
    df = pd.read_csv(DATA_DIR / f"{nombre}.csv")
    return df.drop(columns=TARGET), df[TARGET]


def firma_duplicados(X):
    """Id de grupo: filas con las 28 variables idénticas comparten firma.

    Se usa para que los duplicados exactos nunca queden separados entre train y
    validación/test (Group K-Fold, clase 11)."""
    return pd.util.hash_pandas_object(X[FEATURES_ORIGINALES], index=False).to_numpy()


# ---------------------------------------------------------------- features
def agregar_features(X):
    """Variables nuevas (reglas fila a fila, no aprenden nada -> sin leakage)."""
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
    # ids como categoría (el número no tiene orden). -1 = faltante
    X["agent"] = X["orchestration_agent_id"].fillna(-1).astype(int).astype(str)
    X["org"] = X["organization_id"].fillna(-1).astype(int).astype(str)
    return X


NUM_NUEVAS = ["total_gpu_hours", "total_processes", "gpu_changed", "cost_total",
              "cost_per_process", "weekday", "dayofyear", "has_org", "has_agent",
              "prev_cancel_ratio"]
CAT_NUEVAS = ["agent", "org"]

SETS = {
    "Original (28)": {"num": NUM_ORIGINALES, "cat": CAT_ORIGINALES},
    "FE": {
        "num": [c for c in NUM_ORIGINALES if c not in ("orchestration_agent_id", "organization_id")]
               + NUM_NUEVAS,
        "cat": CAT_ORIGINALES + CAT_NUEVAS,
    },
}


def columnas(set_nombre):
    s = SETS[set_nombre]
    return s["num"], s["cat"]


# ---------------------------------------------------------------- preprocesamiento
def crear_preprocesador(set_nombre, tipo="arbol"):
    """ColumnTransformer que se ajusta dentro del Pipeline (sin leakage).

    tipo="arbol": categóricas -> OrdinalEncoder (categorías raras, <20 filas, se agrupan);
                  numéricas sin tocar (los árboles no necesitan escalado y HGB maneja NaN).
                  Las categóricas quedan PRIMERO -> índices 0..n_cat-1 para HGB.
    tipo="lineal": numéricas -> imputación por mediana + z-score; categóricas -> One-Hot.
    """
    num, cat = columnas(set_nombre)
    if tipo == "arbol":
        ct = ColumnTransformer(
            [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                    encoded_missing_value=-1, min_frequency=20), cat),
             ("num", "passthrough", num)],
            verbose_feature_names_out=False,
        )
    else:
        ct = ColumnTransformer(
            [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
             ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                   OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                 min_frequency=20, sparse_output=False)), cat)],
            verbose_feature_names_out=False,
        )
    return ct


def idx_categoricas(set_nombre):
    """Índices de las categóricas a la salida del preprocesador 'arbol' (para HGB)."""
    return list(range(len(columnas(set_nombre)[1])))


def pipeline(set_nombre, modelo, tipo="arbol"):
    return make_pipeline(crear_preprocesador(set_nombre, tipo), modelo)


# ---------------------------------------------------------------- validación
SCORING = {"bal_acc": "balanced_accuracy", "roc_auc": "roc_auc"}
N_FOLDS = 5


def crear_cv(esquema="grupo", n_splits=N_FOLDS):
    """'grupo': StratifiedGroupKFold (duplicados nunca separados) -> escenario reservas nuevas.
    'aleatorio': StratifiedKFold por fila -> escenario test con el mismo origen (puede repetir filas)."""
    if esquema == "grupo":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def evaluar_cv(modelo, X, y, grupos, esquema="grupo", n_jobs=5):
    """k-fold CV solo sobre train. Devuelve medias y desvíos de train y validación."""
    grupos = grupos if esquema == "grupo" else None  # StratifiedKFold no usa grupos
    r = cross_validate(modelo, X, y, groups=grupos, cv=crear_cv(esquema), scoring=SCORING,
                       return_train_score=True, n_jobs=n_jobs)
    out = {}
    for m in SCORING:
        out[f"val_{m}"] = r[f"test_{m}"].mean()
        out[f"val_{m}_std"] = r[f"test_{m}"].std()
        out[f"train_{m}"] = r[f"train_{m}"].mean()
    out["brecha_bal_acc"] = out["train_bal_acc"] - out["val_bal_acc"]
    out["seg"] = r["fit_time"].mean()
    return out


def metricas(y, proba, umbral=0.5):
    pred = (proba >= umbral).astype(int)
    return {"bal_acc": balanced_accuracy_score(y, pred), "roc_auc": roc_auc_score(y, proba)}


def mejor_umbral(y, proba, grilla=np.arange(0.20, 0.801, 0.01)):
    """Umbral que maximiza balanced accuracy (se busca SOLO con predicciones OOF de train)."""
    ba = [balanced_accuracy_score(y, proba >= t) for t in grilla]
    i = int(np.argmax(ba))
    return float(grilla[i]), float(ba[i])


# ---------------------------------------------------------------- análisis por cohortes
def tabla_cohorte(df, col, y_col="y", pred_col="pred"):
    """Por cada valor de `col`: tamaño, % del total, tasa real de cancelación,
    balanced accuracy, FP, FN y % de los errores totales que aporta."""
    err_tot = (df[y_col] != df[pred_col]).sum()
    filas = []
    for v, g in df.groupby(col, dropna=False):
        yv, pv = g[y_col], g[pred_col]
        fp = int(((pv == 1) & (yv == 0)).sum())
        fn = int(((pv == 0) & (yv == 1)).sum())
        ba = balanced_accuracy_score(yv, pv) if yv.nunique() == 2 else np.nan
        filas.append({col: v, "n": len(g), "%_datos": 100 * len(g) / len(df),
                      "tasa_cancel": yv.mean(), "bal_acc": ba, "error_%": 100 * (yv != pv).mean(),
                      "FP": fp, "FN": fn, "%_de_errores": 100 * (fp + fn) / err_tot})
    return pd.DataFrame(filas).sort_values("n", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- figuras
def guardar(fig, nombre):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / nombre, dpi=130, bbox_inches="tight")


plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
