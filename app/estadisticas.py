# app/estadisticas.py
from dash import html, dcc, dash_table
import pandas as pd
from pathlib import Path
import plotly.express as px

def _load_metrics():
    t1 = Path("experiments/t1/metrics_t1.csv")
    t2 = Path("experiments/t2/metrics_t2.csv")
    df1 = pd.read_csv(t1, index_col=0) if t1.exists() else pd.DataFrame()
    df2 = pd.read_csv(t2, index_col=0) if t2.exists() else pd.DataFrame()
    return df1, df2

def _bar_from_metrics(df: pd.DataFrame, titulo: str):
    if df.empty:
        return px.scatter(title=f"Sin métricas para {titulo}")
    dfp = df.reset_index().melt(id_vars="index", var_name="modelo", value_name="valor")
    dfp = dfp.rename(columns={"index": "metrica"})
    return px.bar(dfp, x="modelo", y="valor", color="metrica", barmode="group", title=titulo)

def layout_estadisticas():
    df1, df2 = _load_metrics()
    fig1 = _bar_from_metrics(df1, "T1 – Riesgo próximo semestre")
    fig2 = _bar_from_metrics(df2, "T2 – Deserción ≤ 1 año")

    return html.Div(
        [
            html.Div(
                [
                    html.H4("T1 – Métricas"),
                    dcc.Graph(figure=fig1, id="graf-t1"),
                    dash_table.DataTable(
                        id="tabla-t1",
                        data=(df1.reset_index().to_dict("records") if not df1.empty else []),
                        columns=([{"name": c, "id": c} for c in (["index"] + df1.columns.tolist())] if not df1.empty else []),
                        style_table={"overflowX": "auto"},
                        page_size=10,
                    ),
                ],
                style={"marginBottom": "24px"},
            ),
            html.Div(
                [
                    html.H4("T2 – Métricas"),
                    dcc.Graph(figure=fig2, id="graf-t2"),
                    dash_table.DataTable(
                        id="tabla-t2",
                        data=(df2.reset_index().to_dict("records") if not df2.empty else []),
                        columns=([{"name": c, "id": c} for c in (["index"] + df2.columns.tolist())] if not df2.empty else []),
                        style_table={"overflowX": "auto"},
                        page_size=10,
                    ),
                ]
            ),
        ]
    )