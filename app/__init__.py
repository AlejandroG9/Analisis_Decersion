# app/__init__.py
import dash, diskcache
from pathlib import Path

cache_dir = Path(__file__).with_name(".cache")
bg_mgr = dash.DiskcacheManager(diskcache.Cache(cache_dir))

dash_app = dash.Dash(__name__, background_callback_manager=bg_mgr)  # instancia
app = dash_app           # ← alias con el nombre que esperan los imports

from .layout import layout
app.layout = layout

from . import callbacks