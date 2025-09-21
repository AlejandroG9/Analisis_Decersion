# app/datos.py
from dash import html, dcc

def layout_datos():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Archivo diagnóstico (wide)"),
                            dcc.Dropdown(
                                id="datos-archivo-unico",
                                options=[],   # lo llena el callback
                                value=None,
                                clearable=False,
                                style={"width": "520px"},
                            ),
                        ],
                        style={"marginRight": "24px"},
                    ),
                    html.Div(
                        [
                            html.Label("Agrupar por periodo"),
                            dcc.RadioItems(
                                id="agrupar-por-periodo",
                                options=[
                                    {"label": "No", "value": "no"},
                                    {"label": "Sí", "value": "si"},
                                ],
                                value="si",
                                inline=True,
                            ),
                        ],
                        style={"alignSelf": "flex-end"},
                    ),
                    html.Div(
                        [
                            html.Label("Área para Heatmap"),
                            dcc.Dropdown(
                                id="area-heatmap",
                                options=[
                                    {"label": "Aritmética", "value": "Aritmetica"},
                                    {"label": "Álgebra", "value": "Algebra"},
                                    {"label": "Geometría", "value": "Geometria"},
                                    {"label": "Trigonometría", "value": "Trigonometria"},
                                ],
                                value="Aritmetica",
                                clearable=False,
                                style={"width": "260px"},
                            ),
                        ],
                        style={"marginLeft": "24px"},
                    ),
                ],
                style={"display": "flex", "gap": "12px", "marginBottom": "14px", "flexWrap": "wrap"},
            ),

            # FILA 1 — medias por área y por periodo
            html.Div(
                [
                    dcc.Graph(id="graf-mean-area"),
                    dcc.Graph(id="graf-box-area"),
                ],
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px"},
            ),

            # FILA 2 — histograma global y prevalencia de targets
            html.Div(
                [
                    dcc.Graph(id="graf-hist-global"),
                    dcc.Graph(id="graf-prevalencia"),
                ],
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px", "marginTop": "14px"},
            ),

            # FILA 3 — heatmap por área (dificultad por ítem)
            html.Div(
                [
                    dcc.Graph(id="graf-heatmap-area"),
                ],
                style={"marginTop": "14px"},
            ),
            # FILA 4 — heatmap por área (dificultad por ítem)
            # ... dentro de layout_datos(), al final o donde prefieras:
            html.Div(
                [
                    dcc.Graph(id="graf-facet-area"),
                ],
                style={"marginTop": "14px"},
            ),
            # FILA 5 — distplot por "Término"
            html.Div(
                [
                    dcc.Graph(id="graf-dist-termino"),
                ],
                style={"marginTop": "14px"},
            ),
            # FILA 6 — Exploración enlazada: Scatter ↔ Parallel Categories
            html.Div(
                [
                    dcc.Store(id="selected-idx-parcats", data=[]),  # estado de selección compartida
                    html.H4("Exploración enlazada: Calificaciones ↔ Aprobaciones/Períodos"),
                    html.Div(
                        [
                            dcc.Graph(id="graf-linked-scatter"),
                            dcc.Graph(id="graf-linked-parcats"),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px"},
                    ),
                ],
                style={"marginTop": "18px"},
            ),
            html.Div(
                [
                    html.H4("Alumnos seleccionados"),
                    dcc.Loading(
                        id="loading-seleccion",
                        type="default",
                        children=[
                            html.Div(id="tabla-seleccionados")
                        ]
                    ),
                ],
                style={"marginTop": "20px"},
            ),
            html.Div(
                [
                    html.H4("Scatter independiente"),
                    dcc.Graph(id="graf-scatter-independiente"),
                ],
                style={"marginTop": "20px"},
            ),
            html.Div(
                [
                    html.H4("Scatter 3D independiente"),
                    dcc.Graph(id="graf-scatter-3d"),
                ],
                style={"marginTop": "20px"},
            ),

            # --- Correlaciones ---
            html.Div(
                [
                    html.H4("Pruebas de correlación"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Método"),
                                    dcc.RadioItems(
                                        id="corr-metodo",
                                        options=[
                                            {"label": "Pearson", "value": "pearson"},
                                            {"label": "Spearman", "value": "spearman"},
                                            {"label": "Kendall", "value": "kendall"},
                                        ],
                                        value="pearson",
                                        inline=True,
                                    ),
                                ],
                                style={"marginRight": "24px"},
                            ),
                            html.Div(
                                [
                                    html.Label("Población"),
                                    dcc.Checklist(
                                        id="corr-usar-seleccion",
                                        options=[{"label": "Usar selección (scatter/parcats)", "value": "sel"}],
                                        value=[],  # vacío = usa todo el dataset
                                        inline=True,
                                    ),
                                ]
                            ),
                        ],
                        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "center",
                               "marginBottom": "10px"},
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="graf-corr-heatmap"),
                        ],
                        style={"marginTop": "6px"},
                    ),
                    html.Div(
                        [
                            html.H5("Pares más correlacionados"),
                            html.Div(id="tabla-corr-pares"),
                        ],
                        style={"marginTop": "8px"},
                    ),
                ],
                style={"marginTop": "24px"},
            ),
            # FILA X — Categorías paralelas (Parcats)
            html.Div(
                [
                    html.H4("Flujo por categorías (Parcats)"),
                    dcc.Graph(id="graf-categorias-paralelas"),
                ],
                style={"marginTop": "14px"},
            ),
        ]
    )