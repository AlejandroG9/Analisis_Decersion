import pandas as pd
import numpy as np
import plotly.figure_factory as ff


datos = pd.read_csv("data/raw/diagnostico_wide_new.csv")


##--------Distribucion de Calificaciones------------
x1 = datos[datos['Termino 2024-3'] == "SI"]['Cal Final']
x2 = datos[datos['Termino 2024-3'] == "NO"]['Cal Final']
x1 = x1
x2 = x2
group_labels = ['Si Termino', 'No Termino']

colors = ['slategray', 'magenta']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot([x1, x2], group_labels, bin_size=5,
                         curve_type='normal', # override default 'kde'
                         colors=colors)

# Add title
fig.update_layout(title_text='Distplot with Normal Distribution')
fig.show()


#--------Poscentaje de alumnos aun inscritos 2025-1-----------


import plotly.express as px
df = datos.groupby(['Termino 2025-1']).count().reset_index()
fig = px.pie(df, values='alumno_id', names='Termino 2025-1', color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_layout(title_text='Numero de Alumnos que presentaron y terminaron 2025-1')
fig.show()

#--------Poscentaje de alumnos aun inscritos 2024-3-----------


import plotly.express as px
df = datos.groupby(['Termino 2024-3']).count().reset_index()
fig = px.pie(df, values='alumno_id', names='Termino 2024-3', color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_layout(title_text='Numero de Alumnos que presentaron y terminaron 2024-3')
fig.show()

#------Grafica de Categorias ----------

import plotly.express as px

df_cat = datos
df_cat['Calificacion 2024-3 Round'] = df_cat['Calificacion 2024-3'].round(decimals=0)
fig = px.parallel_categories(df_cat, dimensions=['Aprobado Aritmetica', 'Aprobado Algebra', 'Aprobado Geometria', 'Aprobado General', 'Termino 2024-3', 'Termino 2025-1'],
                color="Cal Final", color_continuous_scale=px.colors.sequential.Inferno,
                labels={'Aprobado Aritmetica':'Aritmetica', 'Aprobado Algebra': 'Algebra', 'Aprobado Geometria': 'Geometria', 'Aprobado General': 'General', 'Termino 2024-3':'Termino 2024-3', 'Termino 2025-1': 'Termino 2025-1'})
fig.show()


# app_linked_parcats.py
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go

# ====== Datos ======
df = datos.copy()  # <- usa tu DF existente
cat_cols = [
    'Aprobado Aritmetica',
    'Aprobado Algebra',
    'Aprobado Geometria',
    'Aprobado General',
    'Termino 2024-3',
    'Termino 2025-1'
]
for c in cat_cols:
    df[c] = df[c].astype(str).fillna('NA')

x_col = 'Calificacion 2024-3'
y_col = 'Cal Final'
df = df.reset_index(drop=False).rename(columns={'index':'_idx'})  # índice estable

colorscale = [[0, 'gray'], [1, 'firebrick']]
base_color = np.zeros(len(df), dtype='uint8')

def make_scatter(selected_idx=None):
    sel_mask = np.zeros(len(df), dtype=bool)
    if selected_idx:
        sel_mask[selected_idx] = True
    marker_color = np.where(sel_mask, 'firebrick', 'gray')
    opacity = np.where(sel_mask, 1.0, 0.25)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[y_col],
        mode='markers',
        customdata=df[['_idx']].to_numpy(),
        marker={'color': marker_color, 'size': 7, 'opacity': opacity},
        selected={'marker': {'color': 'firebrick', 'size': 9}},
        unselected={'marker': {'opacity': 0.25}}
    ))
    fig.update_layout(
        height=450, margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title=x_col, yaxis_title=y_col, dragmode='lasso'
    )
    return fig

def make_parcats(selected_idx=None):
    color = np.zeros(len(df), dtype='uint8')
    if selected_idx:
        color[selected_idx] = 1
    dimensions = [dict(values=df[c], label=c) for c in cat_cols]
    fig = go.Figure()
    fig.add_trace(go.Parcats(
        domain={'y': [0, 1]},
        dimensions=dimensions,
        line={'colorscale': colorscale, 'cmin': 0, 'cmax': 1, 'color': color, 'shape': 'hspline'},
        bundlecolors=True,
        labelfont={'size': 12},
        tickfont={'size': 10}
    ))
    fig.update_layout(height=350, margin=dict(l=40, r=20, t=10, b=10))
    return fig

app = Dash(__name__)
app.layout = html.Div([
    html.H3("Exploración enlazada: Scatter ↔ Parallel Categories"),
    dcc.Graph(id='scatter', figure=make_scatter()),
    dcc.Graph(id='parcats', figure=make_parcats()),
    dcc.Store(id='selected-idx', data=[])  # guardamos índices seleccionados
])

# 1) Selección en Scatter -> actualiza ambos gráficos
@app.callback(
    Output('selected-idx', 'data'),
    Input('scatter', 'selectedData'),
    prevent_initial_call=True
)
def scatter_select(selectedData):
    if not selectedData or 'points' not in selectedData:
        return []
    # Recupera _idx desde customdata
    idx = [p['customdata'][0] for p in selectedData['points']]
    return idx

# 2) Click en Parcats -> actualiza selección (usa clickData)
@app.callback(
    Output('selected-idx', 'data'),
    Input('parcats', 'clickData'),
    State('selected-idx', 'data'),
    prevent_initial_call=True
)
def parcats_click(clickData, current):
    # En parcats, clickData puede traer 'pointNumber' o 'pointNumbers'
    if not clickData:
        return no_update
    pts = clickData.get('points', [])
    if not pts:
        return no_update
    # pointNumbers son índices de filas subyacentes
    # Si no aparece, dejamos como está
    pnums = pts[0].get('pointNumbers')
    if pnums is None:
        pn = pts[0].get('pointNumber')
        pnums = [pn] if pn is not None else None
    if pnums is None:
        return no_update
    return pnums

# 3) Redibuja figuras cuando cambia la lista de seleccionados
@app.callback(
    Output('scatter', 'figure'),
    Output('parcats', 'figure'),
    Input('selected-idx', 'data')
)
def update_figures(selected_idx):
    return make_scatter(selected_idx), make_parcats(selected_idx)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8056)

