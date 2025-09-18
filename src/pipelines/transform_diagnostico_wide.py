import pandas as pd
import re
from pathlib import Path
from src.config.settings import load_settings

def main():
    cfg = load_settings()
    raw = Path(cfg["data"]["raw_dir"])
    processed = Path(cfg["data"]["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)

    # === 1. Cargar dataset wide ===
    df = pd.read_csv(raw / "diagnostico_wide.csv")

    # Identificar columnas de ítems (terminan en número)
    item_cols = [c for c in df.columns if re.search(r"\d+$", c)]
    meta_cols = ["alumno_id", "grupo", "carrera"]

    # === 2. Wide → Long (1 fila por respuesta) ===
    df_long = df.melt(
        id_vars=meta_cols,
        value_vars=item_cols,
        var_name="item_id",
        value_name="correcto"
    )

    # normalizar respuestas a 0/1
    df_long["correcto"] = df_long["correcto"].astype(int).clip(0, 1)

    # extraer tema (Aritmetica, Algebra, Geometria, Trigonometria)
    df_long["tema"] = df_long["item_id"].str.extract(r"^([A-Za-z]+)")[0]

    # === 3. Features por alumno y tema ===
    tema_pct = df_long.groupby(["alumno_id", "grupo", "tema"])["correcto"].mean().unstack("tema")
    tema_pct = tema_pct.add_prefix("pct_")  # ej. pct_Aritmetica

    # % correcto global
    pct_global = df_long.groupby(["alumno_id", "grupo"])["correcto"].mean().rename("pct_correcto")

    # === 4. Merge final ===
    features = pct_global.reset_index().merge(tema_pct.reset_index(), on=["alumno_id", "grupo"], how="left")

    # Guardar
    out_path = processed / "diagnostico_features_items.csv"
    features.to_csv(out_path, index=False, encoding="utf-8")
    print(f"OK → {out_path} ({features.shape[0]} filas, {features.shape[1]} columnas)")

if __name__ == "__main__":
    main()