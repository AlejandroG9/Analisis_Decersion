import os
import json
import time
import shutil
import datetime
import numpy as np
import pandas as pd
from joblib import dump
from src.metodos.ProcesadorDeDatos import cargar_datos
from src.metodos.BoostedTreesV2GridAuto import ExperimentacionBoostedTrees
from src.metodos.RandomForestV2GridAuto import ExperimentacionRandomForest
import plotly.express as px

def save_model_metadata(path_joblib: str, model, algoritmo: str,
                        experimento: str, run_id: str, rep: int, pct_train: float,
                        extra: dict | None = None):
    """
    Guarda metadatos junto al .joblib en un .json del mismo nombre.
    """
    meta = {
        "algoritmo": algoritmo,
        "experimento": experimento,
        "run_id": run_id,
        "replica": rep,
        "pct_train": pct_train,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_features": getattr(model, "n_features_in_", None),
        "params": getattr(model, "get_params", lambda: {})(),
    }
    if extra:
        meta.update(extra)
    path_json = path_joblib.replace(".joblib", ".json")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def point_latest_symlink(src_dir: str, latest_link: str):
    """
    Apunta `latest_link` (symlink) hacia `src_dir`. Si el symlink no es posible (p.ej. Windows sin permisos),
    intenta copiar (vacío) o ignora silenciosamente.
    """
    try:
        # si ya existe, elimínalo
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        elif os.path.exists(latest_link):
            # si existe como carpeta normal no la borramos; usa otro nombre o gestiona según tu política
            pass
        # crea symlink
        os.symlink(src_dir, latest_link)
    except Exception:
        # fallback opcional: crea carpeta y deja un README con el destino real
        os.makedirs(latest_link, exist_ok=True)
        with open(os.path.join(latest_link, "README.txt"), "w", encoding="utf-8") as f:
            f.write(f"Última corrida está en: {src_dir}\n")


def build_artifact_index(models_root: str, run_id: str, out_csv: str):
    """
    Escanea modelos .joblib bajo `models_root/*/<run_id>/*.joblib` y escribe un índice CSV.
    """
    rows = []
    for exp_name in os.listdir(models_root):
        exp_dir = os.path.join(models_root, exp_name, run_id)
        if not os.path.isdir(exp_dir):
            continue
        for fname in os.listdir(exp_dir):
            if not fname.endswith(".joblib"):
                continue
            algo = "rf" if fname.startswith("rf_") else ("bt" if fname.startswith("bt_") else "unknown")
            rows.append({
                "Experimento": exp_name,
                "run_id": run_id,
                "algoritmo": algo,
                "archivo": fname,
                "ruta": os.path.join(exp_dir, fname),
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)



# ---------------- Paths base ----------------
TASK = "t1"  # cámbialo a "t2" si quieres otra tarea
base_exp_root = os.path.join("experiments", TASK)
base_models_root = os.path.join("models", TASK)
base_metrics = os.path.join("results", "metrics")
base_figs = os.path.join("results", "figures")

for p in [base_exp_root, base_models_root, base_metrics, base_figs]:
    os.makedirs(p, exist_ok=True)

# ---------------- Config ----------------
dir_path = "data/raw"
archivo_v5 = "diagnostico_wide_new.csv"

pe = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
replicas = 60
SEMILLA = 123

columnas_eliminar_v5 = [
    'alumno_id', 'grupo', 'carrera', 'Curso',
    'No_Materias 2024-3','Calificacion 2024-3',
    'No_Materias 2025-1','Calificacion 2025-1',
    'Inscrito 2024-3','Inscrito  2025-1','Termino 2025-1',
    'M_Reprobadas 2024-3','M_Reprobadas 2025-1',
    'M_Aprobadas 2024-3','M_Aprobadas 2025-1',
    'Porcentaje_M_R_2024-3','Porcentaje_M_R_2025-1',
    'Porcentaje_M_A_2024-3','Porcentaje_M_A_2025-1',
    'MAT_No_Materias 2024-3','MAT_Calificacion 2024-3',
    'MAT_No_Materias 2025-1','MAT_Calificacion 2025-1',
    'MAT_Inscrito 2024-3','MAT_Inscrito  2025-1',
    'MAT_M_Reprobadas 2024-3','MAT_M_Reprobadas 2025-1',
    'MAT_M_Aprobadas 2024-3','MAT_M_Aprobadas 2025-1',
    'MAT_Porcentaje_M_R_2024-3','MAT_Porcentaje_M_R_2025-1',
    'MAT_Porcentaje_M_A_2024-3','MAT_Porcentaje_M_A_2025-1'
]

dataset_capt_v5 = cargar_datos(
    archivo=os.path.join(dir_path, archivo_v5),
    indices_eliminar=columnas_eliminar_v5,
    normalizado=False
)
dataset_norm_v5 = cargar_datos(
    archivo=os.path.join(dir_path, archivo_v5),
    indices_eliminar=columnas_eliminar_v5,
    normalizado=True
)

# ---------------- Experimentos ----------------
experimentos = [
    {'Nombre': "Capt_V5", 'DataSet': dataset_capt_v5, 'Respuesta': 'Termino 2024-3', 'ClaseDelModelo': 'Clasificacion'},
    {'Nombre': "Norm_V5", 'DataSet': dataset_norm_v5, 'Respuesta': 'Termino 2024-3', 'ClaseDelModelo': 'Clasificacion'},
]

run_id = time.strftime("%Y%m%d-%H%M%S")  # para distinguir corridas

resumen_rf = []
resumen_bt = []

for exp in experimentos:
    nom = exp["Nombre"]

    # subcarpetas por experimento
    exp_dir = os.path.join(base_exp_root, nom, run_id)
    mdl_dir = os.path.join(base_models_root, nom, run_id)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(mdl_dir, exist_ok=True)

    print(f'==> Inicia Experimentación {nom}')

    # -------- Random Forest --------
    rf_res = ExperimentacionRandomForest(
        DataSet=exp['DataSet'],
        Respuesta=exp['Respuesta'],
        Semilla=SEMILLA,
        pe=pe,
        Replicas=replicas,
        ClaseDelModelo=exp['ClaseDelModelo'],
        Nombre=nom
    )
    # guarda dict de modelos (clave C{rep}S{p}) por separado
    # y logs de errores / predicciones
    rf_error_csv = os.path.join(exp_dir, f'rf_error_{nom}.csv')
    rf_pred_csv  = os.path.join(exp_dir, f'rf_predic_{nom}.csv')
    rf_res["RFSerror"].to_csv(rf_error_csv, index=False)
    rf_res["RFSpredic"].to_csv(rf_pred_csv, index=False)

    # guarda cada modelo entrenado (opcional: uno por réplica/porcentaje)
    for key, model in rf_res["RFS"].items():
        # key es del tipo "C{replica}S{pct}"
        job = os.path.join(mdl_dir, f'rf_{key}.joblib')
        dump(model, job)
        # parsea replica y pct
        try:
            rep = int(key.split("S")[0].removeprefix("C"))
            pct = float(key.split("S")[1])
        except Exception:
            rep, pct = -1, -1.0
        # guarda metadatos JSON
        save_model_metadata(job, model, algoritmo="RandomForest",
                            experimento=nom, run_id=run_id, rep=rep, pct_train=pct)





    # -------- Boosted Trees (XGBoost) --------
    bt_res = ExperimentacionBoostedTrees(
        DataSet=exp['DataSet'],
        Respuesta=exp['Respuesta'],
        Semilla=SEMILLA,
        pe=pe,
        Replicas=replicas,
        ClaseDelModelo=exp['ClaseDelModelo'],
        Nombre=nom
    )
    bt_error_csv = os.path.join(exp_dir, f'bt_error_{nom}.csv')
    bt_pred_csv  = os.path.join(exp_dir, f'bt_predic_{nom}.csv')
    bt_res["BRTSerror"].to_csv(bt_error_csv, index=False)
    bt_res["BRTpredic"].to_csv(bt_pred_csv, index=False)

    for key, model in bt_res["BRTS"].items():
        job = os.path.join(mdl_dir, f'bt_{key}.joblib')
        dump(model, job)
        try:
            rep = int(key.split("S")[0].removeprefix("C"))
            pct = float(key.split("S")[1])
        except Exception:
            rep, pct = -1, -1.0
        save_model_metadata(job, model, algoritmo="XGBoost",
                            experimento=nom, run_id=run_id, rep=rep, pct_train=pct)

# ---------------- Consolidados a results/metrics ----------------
df_rf_all = pd.concat(resumen_rf, ignore_index=True) if resumen_rf else pd.DataFrame()
df_bt_all = pd.concat(resumen_bt, ignore_index=True) if resumen_bt else pd.DataFrame()

if not df_rf_all.empty:
    df_rf_all.to_csv(os.path.join(base_metrics, f"rf_all_metrics_{run_id}.csv"), index=False)
if not df_bt_all.empty:
    df_bt_all.to_csv(os.path.join(base_metrics, f"bt_all_metrics_{run_id}.csv"), index=False)

# ---------------- Figura de ejemplo a results/figures ----------------
if not df_rf_all.empty and "Accuracy" in df_rf_all.columns:
    fig = px.box(df_rf_all, x="PercentTest", y="Accuracy", color="Experimento",
                 title="Accuracy RF por % Entrenamiento")
    # requiere: pip install -U kaleido
    fig.write_image(os.path.join(base_figs, f"rf_accuracy_boxplot_{run_id}.png"))


# --- Actualiza enlaces "latest" por TAREA (global) ---
latest_exp_global = os.path.join(base_exp_root, "latest")
latest_mdl_global = os.path.join(base_models_root, "latest")
point_latest_symlink(os.path.join(base_exp_root, run_id), latest_exp_global)
point_latest_symlink(os.path.join(base_models_root, run_id), latest_mdl_global)

# --- (Opcional) latest por experimento ---
for exp in experimentos:
    nom = exp["Nombre"]
    point_latest_symlink(
        os.path.join(base_exp_root, nom, run_id),
        os.path.join(base_exp_root, nom, "latest")
    )
    point_latest_symlink(
        os.path.join(base_models_root, nom, run_id),
        os.path.join(base_models_root, nom, "latest")
    )


artifact_idx_csv = os.path.join(base_metrics, f"artifact_index_{run_id}.csv")
build_artifact_index(base_models_root, run_id, artifact_idx_csv)

