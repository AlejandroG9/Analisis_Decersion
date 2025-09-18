# app/layout.py
from dash import html, dcc
from .datos import layout_datos
from .estadisticas import layout_estadisticas

layout = html.Div(
    [
        html.H2("Riesgo Académico – Dashboard"),
        dcc.Tabs(
            [
                dcc.Tab(label="Datos para el Modelo", children=layout_datos()),
                dcc.Tab(label="Resultados del Modelo", children=layout_estadisticas()),
                # dcc.Tab(label="Consumo del Modelo", children=layout_consumo()),  # opcional
            ]
        ),
    ],
    style={"fontFamily": "Arial", "padding": "20px"},
)