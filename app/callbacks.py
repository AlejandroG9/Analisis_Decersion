# app/callbacks.py
from dash import Input, Output, callback, State, no_update, ctx
from dash.exceptions import PreventUpdate
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import unicodedata
import re
import plotly.figure_factory as ff

# ========= Config: archivo fijo =========
CSV_PATH = Path("data/raw/diagnostico_wide_new.csv")

AREAS = ["Aritmetica", "Algebra", "Geometria", "Trigonometria"]

# ---------- Helpers de normalización / detección ----------

def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

def _canon(col: str) -> str:
    # normaliza: quita acentos, colapsa espacios/guiones/bajos, trim, lower
    s = _strip_accents(col)
    s = s.replace(".", " ").replace("_", " ").strip()
    s = re.sub(r"\s+", " ", s).lower()
    return s

def _coerce_binary(s: pd.Series) -> pd.Series:
    # 0/1 numérico
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().any():
        return sn.clip(0, 1).fillna(0)
    # mapeos texto → 0/1
    pos = {"aprobado","si","sí","true","verdadero","yes","y","1"}
    neg = {"reprobado","no","false","falso","0"}
    st = s.astype(str).str.strip().str.lower()
    return st.apply(lambda x: 1 if x in pos else (0 if x in neg else np.nan)).fillna(0)

def _pick_item_columns(df: pd.DataFrame):
    """
    Detecta columnas tipo: Aritmética 1, Aritmetica1, Algebra_2, Geometria03, Trigonometria 4, etc.
    Devuelve dict {AreaEstandar: [columnas originales]}.
    """
    out = {a: [] for a in AREAS}
    pat = re.compile(r"^(aritmetica|algebra|geometria|trigonometria)\s*(\d+)$")
    for c in df.columns:
        cn = _canon(c)
        m = pat.match(cn)
        if m:
            area_key = m.group(1)
            if area_key == "aritmetica":
                out["Aritmetica"].append(c)
            elif area_key == "algebra":
                out["Algebra"].append(c)
            elif area_key == "geometria":
                out["Geometria"].append(c)
            elif area_key == "trigonometria":
                out["Trigonometria"].append(c)
    return {k: v for k, v in out.items() if v}

def _pick_area_cal_cols(df: pd.DataFrame):
    """
    Fallback: detecta 'Cal Aritmetica', 'Cal Algebra', 'Cal Geometria', 'Cal Trigonometria', 'Cal Final'.
    Retorna (area_map, col_final) donde area_map = {'Aritmetica': 'Cal Aritmetica', ...}
    """
    canon_cols = { _canon(c): c for c in df.columns }
    area_map = {}
    for area in ["aritmetica","algebra","geometria","trigonometria"]:
        key = f"cal {area}"
        if key in canon_cols:
            area_map[area.capitalize()] = canon_cols[key]
    col_final = canon_cols["cal final"] if "cal final" in canon_cols else None
    return area_map, col_final

def _build_features(df: pd.DataFrame):
    """
    Si hay columnas por ítem → calcula pct_correcto y pct_Área desde ítems (0..1).
    Si no hay ítems pero sí 'Cal *' → usa esas (0..100 → 0..1).
    Devuelve (feats, dif_por_area):
      feats: DataFrame con columnas ['pct_correcto','pct_Aritmetica',...]
      dif_por_area: dict Area -> DataFrame (dificultad por ítem), vacío si no hay nivel ítem.
    """
    # columnas meta opcionales (no indispensables para gráficos)
    meta_cols = [c for c in ["alumno_id","periodo","grupo","carrera"] if c in df.columns]

    # Intento A: ítems
    items = _pick_item_columns(df)
    if items:
        for cols in items.values():
            for c in cols:
                df[c] = _coerce_binary(df[c])

        feats = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)
        all_items = [c for cols in items.values() for c in cols]
        feats["pct_correcto"] = df[all_items].mean(axis=1)
        for area, cols in items.items():
            feats[f"pct_{area}"] = df[cols].mean(axis=1)

        dif_por_area = {}
        if "periodo" in df.columns:
            for area, cols in items.items():
                tmp = df.groupby("periodo")[cols].mean().T  # filas=ítems, cols=periodos
                tmp.index.name = "item"
                dif_por_area[area] = tmp
        else:
            for area, cols in items.items():
                tmp = pd.DataFrame(df[cols].mean(), columns=["global"])
                tmp.index.name = "item"
                dif_por_area[area] = tmp

        if feats.empty and meta_cols:
            feats = df[meta_cols].copy().join(feats, how="left")
        return feats, dif_por_area

    # Intento B: columnas 'Cal *'
    area_map, col_final = _pick_area_cal_cols(df)
    if area_map:
        feats = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)
        for area, col in area_map.items():
            feats[f"pct_{area}"] = pd.to_numeric(df[col], errors="coerce") / 100.0
        if col_final and col_final in df.columns:
            feats["pct_correcto"] = pd.to_numeric(df[col_final], errors="coerce") / 100.0
        else:
            area_cols = [f"pct_{a}" for a in area_map.keys()]
            feats["pct_correcto"] = feats[area_cols].mean(axis=1)
        return feats, {}

    # Sin nada reconocible
    return pd.DataFrame(), {}

def _ensure_parcats_df(df: pd.DataFrame):
    cat_cols_target = [
        "Aprobado Aritmetica",
        "Aprobado Algebra",
        "Aprobado Geometria",
        "Aprobado Trigonometria",
        "Aprobado General",
        "Termino 2024-3",
        "Termino 2025-1",
    ]
    dfc = df.copy()

    # Asegura columnas categóricas
    for c in cat_cols_target:
        if c not in dfc.columns:
            dfc[c] = "NA"
        dfc[c] = dfc[c].astype(str).fillna("NA")

    # X/Y preferidas (ajusta si quieres otras)
    x_col = "Calificacion 2024-3" if "Calificacion 2024-3" in dfc.columns else None
    y_col = "Calificacion 2025-1" if "Calificacion 2025-1" in dfc.columns else None

    # Fallback via _build_features
    if (x_col is None) or (y_col is None):
        feats, _ = _build_features(dfc)
        if y_col is None and "pct_correcto" in feats.columns:
            dfc["pct_correcto"] = feats["pct_correcto"]
            y_col = "pct_correcto"
        if x_col is None:
            for a in ["pct_Aritmetica", "pct_Algebra", "pct_Geometria", "pct_Trigonometria"]:
                if a in feats.columns:
                    dfc[a] = feats[a]
                    x_col = a
                    break

    # Último fallback: dos numéricas cualesquiera
    if (x_col is None) or (y_col is None):
        num_cols = [c for c in dfc.columns if pd.api.types.is_numeric_dtype(dfc[c])]
        if len(num_cols) >= 2:
            if x_col is None: x_col = num_cols[0]
            if y_col is None: y_col = num_cols[1]
        else:
            if x_col is None:
                dfc["_x_dummy"] = 0.0; x_col = "_x_dummy"
            if y_col is None:
                dfc["_y_dummy"] = 0.0; y_col = "_y_dummy"

    # POSICIONES estables 0..n-1 (sin ordenar/filtrar)
    dfc = dfc.reset_index(drop=True).copy()
    dfc["_pos"] = np.arange(len(dfc), dtype=int)

    return dfc, cat_cols_target, x_col, y_col


def _make_linked_scatter(dfc: pd.DataFrame, x_col: str, y_col: str, selected_pos=None):
    import plotly.graph_objects as go
    n = len(dfc)
    if not selected_pos: selected_pos = []
    sel_mask = np.zeros(n, dtype=bool)
    sel_mask[selected_pos] = True

    marker_color = np.where(sel_mask, "firebrick", "gray")
    opacity = np.where(sel_mask, 1.0, 0.25)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dfc[x_col], y=dfc[y_col],
        mode="markers",
        customdata=dfc[["_pos"]].to_numpy(),  # <-- enviamos POSICIONES
        marker={"color": marker_color, "size": 7, "opacity": opacity},
        selected={"marker": {"color": "firebrick", "size": 9}},
        unselected={"marker": {"opacity": 0.25}},
        selectedpoints=selected_pos if selected_pos else None,
        hovertemplate=f"<b>pos</b>: %{{customdata[0]}}<br><b>{x_col}</b>: %{{x}}<br><b>{y_col}</b>: %{{y}}<extra></extra>"
    ))
    fig.update_layout(
        height=460, margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title=x_col, yaxis_title=y_col, dragmode="lasso",
        uirevision="const"
    )
    return fig


def _make_linked_parcats(dfc: pd.DataFrame, cat_cols: list, selected_pos=None):
    import plotly.graph_objects as go
    n = len(dfc)
    if not selected_pos:
        selected_pos = []

    # 1) Normaliza/estandariza categorías (evita variantes tipo 'sí', 'Si', etc.)
    dfp = dfc.copy()
    def norm_bin(s):
        s = str(s).strip().upper()
        if s in {"APROBADO", "SI", "SÍ", "TRUE", "1"}:
            return "APROBADO" if "APROB" in s or s in {"TRUE","1"} else "SI"
        if s in {"REPROBADO", "NO", "FALSE", "0"}:
            return "REPROBADO" if "REPROB" in s or s in {"FALSE","0"} else "NO"
        return s or "NA"

    # Aplica a columnas por tipo
    for c in cat_cols:
        if "APROBADO" in c.upper():
            dfp[c] = dfp[c].apply(norm_bin).replace({"SI": "APROBADO", "NO": "REPROBADO"})
        elif "TERMINO" in c.upper() or "TÉRMINO" in c.upper():
            dfp[c] = dfp[c].apply(norm_bin).replace({"APROBADO": "SI", "REPROBADO": "NO"})
        else:
            dfp[c] = dfp[c].astype(str).fillna("NA").str.upper()

    # 2) Orden de categorías (coherente en todas las dimensiones)
    order_apr = ["REPROBADO", "APROBADO"]
    order_ter = ["NO", "SI"]

    # 3) Vector de color por posición (como en el FigureWidget)
    color = np.zeros(n, dtype="uint8")
    color[list(map(int, selected_pos))] = 1

    # 4) Etiquetas cortas (quiebra de línea para que no se encimen)
    label_map = {
        "Aprobado Aritmetica": "Aritmética",
        "Aprobado Algebra": "Álgebra",
        "Aprobado Geometria": "Geometría",
        "Aprobado Trigonometria": "Trigonometria",
        "Aprobado General": "General",
        "Termino 2024-3": "Término 2024-3",
        "Termino 2025-1": "Término 2025-1",
    }

    # 5) Construye dimensiones con orden explícito
    dimensions = []
    for c in cat_cols:
        dim = dict(values=dfp[c], label=label_map.get(c, c))
        cu = c.upper()
        if "APROBADO" in cu:
            dim["categoryorder"] = "array"
            dim["categoryarray"] = order_apr
        elif "TERMINO" in cu or "TÉRMINO" in cu:
            dim["categoryorder"] = "array"
            dim["categoryarray"] = order_ter
        else:
            dim["categoryorder"] = "category ascending"
        dimensions.append(dim)

    # 6) Colores: gris translúcido para no “ensuciar” y rojo fuerte para seleccionado
    colorscale = [
        [0.0, "rgba(160,160,160,0.28)"],  # gris claro con alpha
        [1.0, "rgba(178,34,34,1.0)"],     # firebrick sólido
    ]

    TOP = 0.92  # antes 1.0
    BOT = 0.06  # antes 0.0

    fig = go.Figure()
    fig.add_trace(go.Parcats(
        domain={"y": [BOT, TOP]},
        dimensions=dimensions,
        line={
            "colorscale": colorscale,
            "cmin": 0, "cmax": 1,
            "color": color.tolist(),  # por posición
            "shape": "hspline",
        },
        bundlecolors=False,             # crítico: respeta color por línea
        labelfont={"size": 13},
        tickfont={"size": 12}
    ))

    fig.update_layout(
        height=420,
        margin=dict(l=40, r=12, t=30, b=12),  # ⬅️ más margen arriba
        font=dict(size=10),  # tamaño base
        uirevision="const"
    )

    # Si las categorías (ticks) se ven muy grandes:
    # dentro del trace ya tenemos:
    # labelfont={"size": 13}, tickfont={"size": 12}
    return fig


# =================== Callback principal ===================

@callback(
    Output("graf-mean-area", "figure"),
    Output("graf-box-area", "figure"),
    Output("graf-hist-global", "figure"),
    Output("graf-prevalencia", "figure"),
    Output("graf-heatmap-area", "figure"),
    Output("graf-facet-area", "figure"),  # 👈 nuevo
    Output("graf-dist-termino", "figure"),  # 👈 nuevo
    Input("agrupar-por-periodo", "value"),
    Input("area-heatmap", "value"),
)
def _actualizar_graficas(agrupar, area_sel):
    # 1) Existe archivo
    if not CSV_PATH.exists():
        # figuras “vacías” amigables
        fig_empty = px.scatter(title=f"No se encontró {CSV_PATH}")
        fig_empty.update_layout(uirevision="const")
        return fig_empty, fig_empty, fig_empty, fig_empty, fig_empty

    # 2) Cargar
    df = pd.read_csv(CSV_PATH)

    # 3) Construir features
    feats, dif_por_area = _build_features(df)
    if feats.empty:
        msg = "El archivo no tiene columnas por ítem ni 'Cal *' detectables"
        fig_empty = px.scatter(title=msg)
        fig_empty.update_layout(uirevision="const")
        return fig_empty, fig_empty, fig_empty, fig_empty, fig_empty

    # -------- Graf 1: promedio por área --------
    cols_area = [c for c in feats.columns if c.startswith("pct_") and c != "pct_correcto"]
    if cols_area:
        if agrupar == "si" and "periodo" in feats.columns:
            d1 = feats.groupby("periodo")[cols_area].mean().reset_index().melt(
                id_vars="periodo", var_name="area", value_name="valor"
            )
            d1["area"] = d1["area"].str.replace("pct_", "", regex=False)
            fig1 = px.bar(d1, x="area", y="valor", color="periodo", barmode="group",
                          title="Promedio de aciertos por área")
        else:
            d1 = feats[cols_area].mean().reset_index()
            d1.columns = ["area", "valor"]
            d1["area"] = d1["area"].str.replace("pct_", "", regex=False)
            fig1 = px.bar(d1, x="area", y="valor", title="Promedio de aciertos por área")
    else:
        d1 = pd.DataFrame({"area": ["Global"], "valor": [feats["pct_correcto"].mean()]})
        fig1 = px.bar(d1, x="area", y="valor", title="Promedio global de aciertos")

    fig1.update_yaxes(range=[0, 1], fixedrange=True, tickformat=".0%")
    fig1.update_traces(texttemplate="%{y:.1%}", textposition="inside", insidetextanchor="middle", cliponaxis=True)
    fig1.update_layout(uirevision="const", yaxis_title="Proporción", xaxis_title="Área")

    # -------- Graf 2: boxplot por área --------
    if cols_area:
        d2 = feats[cols_area].rename(columns=lambda c: c.replace("pct_", ""))
        d2 = d2.melt(var_name="Área", value_name="Proporción")
        fig2 = px.box(d2, x="Área", y="Proporción", points="suspectedoutliers", title="Distribución por área")
        fig2.update_yaxes(range=[0, 1], tickformat=".0%", fixedrange=True)
    else:
        fig2 = px.box(title="Sin columnas por área (solo global)")
    fig2.update_layout(uirevision="const")

    # -------- Graf 3: histograma global --------
    fig3 = px.histogram(feats, x="pct_correcto", nbins=20, title="Distribución del porcentaje global de aciertos")
    fig3.update_xaxes(range=[0, 1], tickformat=".0%", title="Proporción")
    fig3.update_yaxes(title="Alumnos", fixedrange=True)
    fig3.update_layout(uirevision="const")


    # -------- Graf 4: prevalencia T1/T2 (si están en el CSV) --------
    prev_vals = {}
    for t in ["T1", "T2"]:
        if t in df.columns:
            prev_vals[t] = float(_coerce_binary(df[t]).mean())
    if prev_vals:
        d4 = pd.DataFrame({"target": list(prev_vals.keys()), "prevalencia": list(prev_vals.values())})
        fig4 = px.bar(d4, x="target", y="prevalencia", text="prevalencia", title="Prevalencia de Targets")
        fig4.update_traces(texttemplate="%{y:.1%}", textposition="inside", insidetextanchor="middle", cliponaxis=True)
        fig4.update_yaxes(range=[0, 1], fixedrange=True, tickformat=".0%")
        fig4.update_layout(uirevision="const", yaxis_title="Proporción", xaxis_title="")
    else:
        fig4 = px.scatter(title="No hay columnas T1/T2 en el archivo")
        fig4.update_yaxes(range=[0, 1], fixedrange=True)
        fig4.update_layout(uirevision="const")

    # -------- Graf 5: heatmap por área (si hay ítems) --------
    if dif_por_area and (area_sel in dif_por_area) and not dif_por_area[area_sel].empty:
        mat = dif_por_area[area_sel]
        fig5 = px.imshow(
            mat, aspect="auto", origin="lower", color_continuous_scale="Blues",
            title=f"Dificultad por ítem – {area_sel} (proporción de aciertos)"
        )
        fig5.update_coloraxes(cmin=0, cmax=1, colorbar_title="Aciertos")
        fig5.update_layout(uirevision="const")
    else:
        fig5 = px.imshow(np.zeros((1, 1)), title=f"Sin ítems para {area_sel} o no disponibles en este archivo")
        fig5.update_layout(uirevision="const")


    # -------- Graf 6: scatter facetado por área (tipo tips) --------
    items = _pick_item_columns(df)  # reutilizamos el detector de columnas de ítems
    if not items:
        fig6 = px.scatter(title="No hay columnas por ítem para construir aciertos por área")
        fig6.update_layout(uirevision="const")
    else:
        # Asegura 0/1
        for cols in items.values():
            for c in cols:
                df[c] = _coerce_binary(df[c])

        # Eje X: alumno_id si existe; si no, índice 1..N
        if "alumno_id" in df.columns:
            x_series = df["alumno_id"].astype(str)
        else:
            x_series = pd.Series(np.arange(1, len(df)+1), name="alumno_idx").astype(str)

        # Color: columna "Termino 2025-1" si existe; si no, un valor fijo
        color_col = "Termino 2025-1" if "Termino 2025-1" in df.columns else None
        if color_col:
            color_series = df[color_col].astype(str)
        else:
            color_series = pd.Series(["(sin Termino 2025-1)"]*len(df), name="termino")

        # Armamos DF largo
        long_rows = []
        for area, cols in items.items():
            aciertos = df[cols].sum(axis=1)  # suma de correctos por área
            tmp = pd.DataFrame({
                "alumno_x": x_series.values,
                "aciertos": aciertos.values,
                "area": area,
                "color_var": color_series.values
            })
            long_rows.append(tmp)
        df_long = pd.concat(long_rows, ignore_index=True)

        cat_orders = {"area": [a for a in AREAS if a in df_long["area"].unique().tolist()]}

        fig6 = px.scatter(
            df_long,
            x="alumno_x",
            y="aciertos",
            color="color_var",
            facet_row="area",
            category_orders=cat_orders,
            title="Aciertos por alumno y área",
            render_mode="webgl",
            hover_data={"alumno_x": True, "aciertos": True, "area": True}
        )
        fig6.update_yaxes(title="Aciertos", rangemode="tozero", fixedrange=True)
        fig6.update_xaxes(showticklabels=False)
        fig6.update_layout(
            uirevision="const",
            legend_title_text=color_col if color_col else "Grupo",
            height=1200,
            margin=dict(l=40, r=20, t=60, b=40),
        )

    # -------- Graf 7: distplot por "Término" (comparación de Cal Final) --------
    # Armamos dinámicamente en función de columnas disponibles
    series_list = []
    labels = []

    # Usaremos estas columnas si existen
    col_t243 = "Termino 2024-3"
    col_t251 = "Termino 2025-1"

    def _add_dist_if_possible(mask_col, mask_val, etiqueta):
        if mask_col in df.columns and "Cal Final" in df.columns:
            m = df[mask_col].astype(str).str.upper().str.strip()
            arr = pd.to_numeric(df.loc[m == mask_val, "Cal Final"], errors="coerce").dropna()
            if len(arr) > 0:
                series_list.append(arr)
                labels.append(etiqueta)

    # Ejemplos: “Sí terminó 2024-3”, “No terminó 2025-1”
    _add_dist_if_possible(col_t243, "SI", "Termino 2024-3 = SI")
    _add_dist_if_possible(col_t243, "NO", "Termino 2024-3 = NO")
    _add_dist_if_possible(col_t251, "SI", "Termino 2025-1 = SI")
    _add_dist_if_possible(col_t251, "NO", "Termino 2025-1 = NO")

    if len(series_list) >= 2:
        fig7 = ff.create_distplot(series_list, labels, bin_size=4, show_rug=False)
        fig7.update_layout(title="Distribución de Cal Final por estatus de 'Término'")
        fig7.update_xaxes(title="Cal Final")
        fig7.update_yaxes(title="Densidad", fixedrange=True)
        fig7.update_layout(uirevision="const")
    elif len(series_list) == 1:
        # Con una sola serie, mostramos hist simple para evitar error de distplot
        fig7 = px.histogram(series_list[0], nbins=20, title=f"Distribución de Cal Final ({labels[0]})")
        fig7.update_xaxes(title="Cal Final")
        fig7.update_yaxes(title="Alumnos", fixedrange=True)
        fig7.update_layout(uirevision="const")
    else:
        fig7 = px.scatter(title="No se pudo construir el distplot (faltan columnas 'Termino *' o 'Cal Final')")
        fig7.update_layout(uirevision="const")

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7


# ====== Linked Scatter ↔ Parcats ======

from dash import no_update


from dash import Input, Output, callback, no_update, ctx

@callback(
    Output("selected-idx-parcats", "data"),
    Input("graf-linked-scatter", "selectedData"),
    Input("graf-linked-parcats", "clickData"),
    prevent_initial_call=True,
)
def _update_selection_store(selectedData, clickData):
    trig = ctx.triggered_id

    # A) Selección múltiple en SCATTER → POSICIONES
    if trig == "graf-linked-scatter":
        # 👇 Mantén la selección actual si llegó el segundo evento "vacío"
        if not selectedData or "points" not in selectedData or not selectedData["points"]:
            return no_update

        pos = []
        for p in selectedData["points"]:
            if "customdata" in p and isinstance(p["customdata"], (list, tuple)) and p["customdata"]:
                pos.append(int(p["customdata"][0]))
            elif "pointIndex" in p:
                pos.append(int(p["pointIndex"]))
            elif "pointNumber" in p:
                pos.append(int(p["pointNumber"]))
        return sorted(set(pos))

    # B) Click/selección en PARCATS → POSICIONES (pointNumbers)
    if trig == "graf-linked-parcats":
        if not clickData or not clickData.get("points"):
            return no_update
        p = clickData["points"][0]
        pnums = p.get("pointNumbers")
        if pnums is None:
            pn = p.get("pointNumber")
            pnums = [pn] if pn is not None else []
        return sorted(set(int(x) for x in pnums if x is not None))

    return no_update



@callback(
    Output("graf-linked-scatter", "figure"),
    Output("graf-linked-parcats", "figure"),
    Input("selected-idx-parcats", "data"),    # posiciones seleccionadas
    Input("agrupar-por-periodo", "value"),    # refresco suave/inicial
)
def _render_or_update_linked(selected_pos, _refresh):
    import plotly.express as px
    if not CSV_PATH.exists():
        fig_empty = px.scatter(title=f"No se encontró {CSV_PATH}")
        fig_empty.update_layout(uirevision="const")
        return fig_empty, fig_empty

    df = pd.read_csv(CSV_PATH)
    dfc, cat_cols, x_col, y_col = _ensure_parcats_df(df)

    if not selected_pos:
        selected_pos = []

    fig_scatter = _make_linked_scatter(dfc, x_col, y_col, selected_pos=selected_pos)
    fig_parcats = _make_linked_parcats(dfc, cat_cols, selected_pos=selected_pos)
    return fig_scatter, fig_parcats




from dash import dash_table

@callback(
    Output("tabla-seleccionados", "children"),
    Input("selected-idx-parcats", "data"),
    prevent_initial_call=True
)
def _mostrar_alumnos_seleccionados(idx_list):
    if not idx_list or not CSV_PATH.exists():
        return html.Div("No hay alumnos seleccionados", style={"color": "gray"})

    df = pd.read_csv(CSV_PATH).reset_index(drop=True)

    # filtra por los índices seleccionados
    try:
        df_sel = df.iloc[idx_list].copy()
    except Exception:
        return html.Div("Error al recuperar selección", style={"color": "red"})

    # opcional: muestra solo algunas columnas clave
    cols = [c for c in df_sel.columns if c.lower() in
            {"alumno_id", "nombre","grupo", "carrera", "cal final", "termino 2024-3", "termino 2025-1"}]
    if cols:
        df_sel = df_sel[cols]

    return dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in df_sel.columns],
        data=df_sel.to_dict("records"),
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "padding": "6px"},
        style_header={"backgroundColor": "#f0f0f0", "fontWeight": "bold"},
    )


@callback(
    Output("graf-scatter-independiente", "figure"),
    Input("agrupar-por-periodo", "value"),  # o cualquier otro trigger
)
def _render_scatter_independiente(_):
    if not CSV_PATH.exists():
        fig_empty = px.scatter(title=f"No se encontró {CSV_PATH}")
        fig_empty.update_layout(uirevision="const")
        return fig_empty

    df = pd.read_csv(CSV_PATH)

    # Asegúrate que existan columnas que quieras graficar
    if not {"Calificacion 2024-3", "Cal Final"} <= set(df.columns):
        return px.scatter(title="Faltan columnas 'Calificacion 2024-3' o 'Cal Final'")

    fig = px.scatter(
        df,
        x="MAT_Prom_General",
        y="Asistencia_Curso",
        color="Termino 2025-1" if "Termino 2025-1" in df.columns else None,
        facet_col="carrera",

        title="Scatter independiente: Cal 2024-3 vs Cal Final",
        hover_data=df.columns,
    )
    fig.update_layout(
        uirevision="const",
        height=500,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig



@callback(
    Output("graf-scatter-3d", "figure"),
    Input("agrupar-por-periodo", "value"),  # puedes usar cualquier trigger existente
)
def _render_scatter_3d(_):
    # Aquí usamos un dataset de ejemplo (iris)
    df = pd.read_csv(CSV_PATH)

    fig = px.scatter_3d(
        df,
        x="Prom_General",
        y="MAT_Prom_General",
        z="Cal Final",
        color="Asistencia_Curso",
        symbol="Termino 2025-1",
        opacity=0.6,
        size='Mat_Aprobadas_General', size_max=18,
        title="Scatter 3D de ejemplo (Iris dataset)"
    )

    fig.update_layout(
        height=1000,
        margin=dict(l=40, r=20, t=60, b=40),
        uirevision="const"
    )
    return fig



from dash import dash_table

@callback(
    Output("graf-corr-heatmap", "figure"),
    Output("tabla-corr-pares", "children"),
    Input("corr-metodo", "value"),
    Input("corr-usar-seleccion", "value"),
    Input("selected-idx-parcats", "data"),   # para filtrar cuando el toggle está activo
)
def _render_correlaciones(metodo, usar_sel_flags, selected_idx):
    import plotly.express as px
    import numpy as np
    import pandas as pd

    # 1) Cargar datos
    if not CSV_PATH.exists():
        fig_empty = px.imshow(np.zeros((1,1)), title=f"No se encontró {CSV_PATH}")
        return fig_empty, html.Div("No hay datos", style={"color":"gray"})

    df = pd.read_csv(CSV_PATH).reset_index(drop=True)

    # 2) Filtrar por selección si corresponde
    usar_sel = "sel" in (usar_sel_flags or [])
    if usar_sel and selected_idx:
        # selected_idx son POSICIONES (0..n-1)
        selected_idx = [i for i in selected_idx if 0 <= i < len(df)]
        df = df.iloc[selected_idx].copy()

    # 3) Seleccionar columnas numéricas útiles
    # --- Columnas candidatas numéricas ---
    cols_numericas = [
        # Ítems por área
        *[f"Aritmetica{i}" for i in range(1, 9)],
        *[f"Algebra{i}" for i in range(1, 10)],
        *[f"Geometria{i}" for i in range(1, 9)],
        *[f"Trigonometria{i}" for i in range(1, 5)],

        # Calificaciones
        "Cal Aritmetica", "Cal Algebra", "Cal Geometria", "Cal Trigonometria", "Cal Final",

        # Sumas
        "Suma Aritmetica", "Suma Algebra", "Suma Geometria", "Suma Trigonometria", "Suma Final",

        # Indicadores por periodo
        "No_Materias 2024-3", "Calificacion 2024-3",
        "No_Materias 2025-1", "Calificacion 2025-1",
        "M_Reprobadas 2024-3", "M_Reprobadas 2025-1",
        "M_Aprobadas 2024-3", "M_Aprobadas 2025-1",
        "Porcentaje_M_R_2024-3", "Porcentaje_M_R_2025-1",
        "Porcentaje_M_A_2024-3", "Porcentaje_M_A_2025-1",

        # Indicadores MAT
        "MAT_No_Materias 2024-3", "MAT_Calificacion 2024-3",
        "MAT_No_Materias 2025-1", "MAT_Calificacion 2025-1",
        "MAT_M_Reprobadas 2024-3", "MAT_M_Reprobadas 2025-1",
        "MAT_M_Aprobadas 2024-3", "MAT_M_Aprobadas 2025-1",
        "MAT_Porcentaje_M_R_2024-3", "MAT_Porcentaje_M_R_2025-1",
        "MAT_Porcentaje_M_A_2024-3", "MAT_Porcentaje_M_A_2025-1",

        # Promedios
        "Prom_General", "MAT_Prom_General", "Mat_Aprobadas_General"
    ]

    # filtra solo las columnas presentes en el CSV
    num_df = df[[c for c in cols_numericas if c in df.columns]].apply(pd.to_numeric, errors="coerce")



    # (opcional) excluir columnas "ID" típicas si existen
    for c in ["alumno_id", "id"]:
        if c in num_df.columns:
            num_df = num_df.drop(columns=[c])

    # Quitar columnas constantes o con demasiados NaN
    nunique = num_df.nunique(dropna=True)
    keep_cols = nunique[nunique > 1].index.tolist()
    num_df = num_df[keep_cols].dropna(how="all", axis=1)
    if num_df.shape[1] < 2:
        fig_empty = px.imshow(np.zeros((1,1)), title="No hay suficientes variables numéricas para correlación")
        return fig_empty, html.Div("Variables insuficientes", style={"color": "gray"})

    # 4) Calcular matriz de correlación
    corr = num_df.corr(method=metodo)

    # 5) Heatmap
    fig = px.imshow(
        corr,
        text_auto=False,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        aspect="auto",
        title=f"Matriz de correlación ({metodo.title()})"
    )
    fig.update_layout(
        height=1000,
        margin=dict(l=40, r=20, t=60, b=40),
        uirevision="const",
        coloraxis_colorbar=dict(title="r")
    )
    fig.update_xaxes(side="bottom")

    # 6) Top pares (por |r|) en tabla
    corr_vals = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r):
                corr_vals.append({"Var 1": cols[i], "Var 2": cols[j], "|r|": abs(r), "r": r})

    if not corr_vals:
        tabla = html.Div("No se pudieron calcular pares", style={"color":"gray"})
    else:
        top = pd.DataFrame(corr_vals).sort_values("|r|", ascending=False).head(20)
        # (Opcional) redondeo para visualizar bonito
        top["r"] = top["r"].round(3)
        top["|r|"] = top["|r|"].round(3)
        tabla = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in ["Var 1", "Var 2", "r", "|r|"]],
            data=top.to_dict("records"),
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "6px"},
            style_header={"backgroundColor": "#f0f0f0", "fontWeight": "bold"},
        )

    return fig, tabla