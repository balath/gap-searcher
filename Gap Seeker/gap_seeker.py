import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    ColumnDataSource,
    DataRange1d,
    Div,
    LassoSelectTool,
    Select,
    Paragraph,
    TextInput,
)
from bokeh.plotting import curdoc, figure
from bokeh.models import CustomJS


# ---------------------------------------------------------------------------
# Directorio raíz (parquet y .npy en root)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Espectros: mediana y ejes de longitud de onda (como en spectra_plots.py)
# ---------------------------------------------------------------------------
def _to_array(v):
    """Convierte valor de espectro (lista/array o string) a np.ndarray."""
    if isinstance(v, str):
        v = ast.literal_eval(v)
    if isinstance(v, (list, np.ndarray)) and len(v) > 0:
        return np.array(v, dtype=float)
    return np.array([np.nan], dtype=float)


def median_spectrum(spectra):
    np_spectra = []
    for s_val in spectra.values:
        if isinstance(s_val, (list, np.ndarray)) and len(s_val) > 0:
            np_spectra.append(np.array(s_val, dtype=float))
        elif isinstance(s_val, (list, np.ndarray)) and len(s_val) == 0:
            np_spectra.append(np.array([np.nan], dtype=float))
    if not np_spectra:
        return np.array([np.nan])
    max_len = max(s.shape[0] for s in np_spectra)
    padded_spectra = []
    for s_arr in np_spectra:
        if s_arr.shape[0] < max_len:
            padded_s = np.pad(s_arr, (0, max_len - s_arr.shape[0]), "constant", constant_values=np.nan)
            padded_spectra.append(padded_s)
        else:
            padded_spectra.append(s_arr)
    return np.nanmedian(np.stack(padded_spectra), axis=0)


def load_wavelengths():
    """Carga ejes de longitud de onda desde root (filtered_wl_bp.npy, filtered_wl_rp.npy).
    Para BP se ordena por longitud de onda y se devuelve el índice para reordenar los flujos.
    Devuelve (wl_bp, wl_rp, bp_sort_idx) o (None, None, None)."""
    wl_bp_path = ROOT_DIR / "filtered_wl_bp.npy"
    wl_rp_path = ROOT_DIR / "filtered_wl_rp.npy"
    if not wl_bp_path.is_file():
        wl_bp_path = ROOT_DIR / "wl_bp.npy"
    if not wl_rp_path.is_file():
        wl_rp_path = ROOT_DIR / "wl_rp.npy"
    wl_bp = np.load(wl_bp_path) if wl_bp_path.is_file() else None
    wl_rp = np.load(wl_rp_path) if wl_rp_path.is_file() else None
    bp_sort_idx = None
    if wl_bp is not None:
        bp_sort_idx = np.argsort(wl_bp)
        wl_bp = wl_bp[bp_sort_idx]
    return wl_bp, wl_rp, bp_sort_idx


def load_precomputed_global_medians(wl_bp, wl_rp):
    """Carga medianas globales precalculadas desde spectra_medians.parquet si existe.
    Devuelve (med_bp, med_rp) o (None, None) si no está disponible."""
    med_path = ROOT_DIR / "spectra_medians.parquet"
    if not med_path.is_file():
        return None, None
    try:
        med_df = pd.read_parquet(med_path)
    except Exception:
        return None, None
    if med_df.empty or "kind" not in med_df.columns:
        return None, None
    try:
        row = med_df.loc[med_df["kind"] == "global"].iloc[0]
    except Exception:
        return None, None
    if "bp_median" not in med_df.columns or "rp_median" not in med_df.columns:
        return None, None
    try:
        bp = np.array(row["bp_median"], dtype=float)
        rp = np.array(row["rp_median"], dtype=float)
    except Exception:
        return None, None
    if wl_bp is not None and len(bp) != len(wl_bp):
        bp = bp[: len(wl_bp)] if len(bp) >= len(wl_bp) else np.pad(
            bp.astype(float), (0, len(wl_bp) - len(bp)), "constant", constant_values=np.nan
        )
    if wl_rp is not None and len(rp) != len(wl_rp):
        rp = rp[: len(wl_rp)] if len(rp) >= len(wl_rp) else np.pad(
            rp.astype(float), (0, len(wl_rp) - len(rp)), "constant", constant_values=np.nan
        )
    return bp, rp


def load_precomputed_bin_medians():
    """Carga medianas por bins desde spectra_medians.parquet si existe.
    Devuelve (mg_edges, color_edges, med_dict) o (None, None, {})."""
    med_path = ROOT_DIR / "spectra_medians.parquet"
    if not med_path.is_file():
        return None, None, {}
    try:
        med_df = pd.read_parquet(med_path)
    except Exception:
        return None, None, {}
    if med_df.empty or "kind" not in med_df.columns:
        return None, None, {}

    bins_df = med_df.loc[med_df["kind"] == "bin"].copy()
    if bins_df.empty:
        return None, None, {}

    needed = {"mg_left", "mg_right", "color_left", "color_right", "bp_median", "rp_median"}
    if needed - set(bins_df.columns):
        return None, None, {}

    def _edges_from_lr(left_col: str, right_col: str):
        lefts = pd.to_numeric(bins_df[left_col], errors="coerce").dropna().astype(float).values
        rights = pd.to_numeric(bins_df[right_col], errors="coerce").dropna().astype(float).values
        if lefts.size == 0 or rights.size == 0:
            return None
        edges = np.unique(np.concatenate([lefts, rights]))
        edges = np.sort(edges.astype(float))
        if edges.size < 2:
            return None
        return edges

    mg_edges = _edges_from_lr("mg_left", "mg_right")
    color_edges = _edges_from_lr("color_left", "color_right")
    if mg_edges is None or color_edges is None:
        return None, None, {}

    # Diccionario de medianas: (mg_left, mg_right, color_left, color_right) -> (bp, rp, n)
    med_dict = {}
    for _, r in bins_df.iterrows():
        try:
            key = (float(r["mg_left"]), float(r["mg_right"]), float(r["color_left"]), float(r["color_right"]))
            bp = np.array(r["bp_median"], dtype=float)
            rp = np.array(r["rp_median"], dtype=float)
            n = int(r["n_sources"]) if "n_sources" in bins_df.columns and pd.notna(r.get("n_sources")) else 0
        except Exception:
            continue
        med_dict[key] = (bp, rp, n)

    return mg_edges, color_edges, med_dict


def load_spectra_at_startup():
    """Carga reduced_spectra.parquet y longitudes de onda al inicio. Espectros en memoria por source_id."""
    spectra_path = ROOT_DIR / "reduced_spectra.parquet"
    if not spectra_path.is_file():
        return None, None, None, None, None, None
    df_spec = pd.read_parquet(spectra_path)
    required = {"source_id", "BP", "RP"}
    if required - set(df_spec.columns):
        return None, None, None, None, None, None
    wl_bp, wl_rp, bp_sort_idx = load_wavelengths()
    df_spec = df_spec.set_index("source_id")  # índice por source_id para búsqueda rápida
    if wl_bp is None or wl_rp is None:
        return df_spec, None, None, None, None, None
    # Medianas globales: SIEMPRE precalculadas. No recalcular en caliente sobre todo el catálogo.
    global_med_bp, global_med_rp = load_precomputed_global_medians(wl_bp, wl_rp)
    return df_spec, wl_bp, wl_rp, bp_sort_idx, global_med_bp, global_med_rp


# Carga de espectros y longitudes de onda al iniciar la aplicación (antes de cualquier parquet de dimensiones)
_SPECTRA_DF = None
_wl_bp = None
_wl_rp = None
_bp_sort_idx = None
_global_median_bp = None
_global_median_rp = None
_spectra_load_done = False

# Binificación para visualización por color y magnitud
_BINS_CAMD = 4
_bin_mg_intervals = None
_bin_color_intervals = None

# Diccionarios de ColumnDataSource para espectros binned
_bp_all_bin_sources = {}
_bp_sel_bin_sources = {}
_rp_all_bin_sources = {}
_rp_sel_bin_sources = {}

# Figuras por bins
_bp_bin_figs = []
_rp_bin_figs = []

# Columnas contenedoras en el layout para los bins
bp_bins_column = column()
rp_bins_column = column()

# Bordes de bin precalculados (si existen en spectra_medians.parquet)
_precomputed_mg_edges = None
_precomputed_color_edges = None

# Modo de superposición de medianas de selección en bins:
# - "by_bin": mediana por cada bin (comportamiento actual, por defecto)
# - "global": misma mediana global de la selección en todos los bins (negro)
_selection_medians_mode = "by_bin"
_cached_sel_median_bp = None
_cached_sel_median_rp = None
_cached_sel_bin_bp = {}
_cached_sel_bin_rp = {}

def _ensure_spectra_loaded():
    global _SPECTRA_DF, _wl_bp, _wl_rp, _bp_sort_idx, _global_median_bp, _global_median_rp, _spectra_load_done
    if not _spectra_load_done:
        _spectra_load_done = True
        _SPECTRA_DF, _wl_bp, _wl_rp, _bp_sort_idx, _global_median_bp, _global_median_rp = load_spectra_at_startup()
_ensure_spectra_loaded()

# ---------------------------------------------------------------------------
# Carga de datos (parquet de dimensiones: solo source_id, BP_RP, MG y columnas de dimensiones; sin espectros)
# ---------------------------------------------------------------------------
def load_data_from_path(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"source_id", "BP_RP", "MG"}
    if required - set(df.columns):
        raise ValueError(f"El parquet de dimensiones debe contener {required}. Faltan: {required - set(df.columns)}")
    return df


df = pd.DataFrame(columns=["source_id", "BP_RP", "MG", "z1", "z2"])
dim_cols = ["z1", "z2"]  # placeholders si el parquet no tiene dimensiones aún

# Columnas de dimensiones: las que empiezan por BP_ o RP_
def is_dim_column(name: str) -> bool:
    return name.startswith("BP_") or name.startswith("RP_")
   
# ---------------------------------------------------------------------------
# Fuentes de datos
# ---------------------------------------------------------------------------
# Pairplot: x, y según dimensiones elegidas; índice = fila en df
pair_x = df[dim_cols[0]].values
pair_y = df[dim_cols[1]].values
# source_id como string para evitar BokehUserWarning (JS no representa bien enteros > 2^53)
pairplot_source = ColumnDataSource(data={
    "x": pair_x,
    "y": pair_y,
    "source_id": df["source_id"].astype(str).values,
})

# ---------------------------------------------------------------------------
# Persistencia de selección (por source_id, no por índices)
# ---------------------------------------------------------------------------
# Guardamos la selección como conjunto de source_id (strings). Esto permite
# mantener la selección aunque cambien dimensiones o incluso se cargue otro parquet,
# siempre que esos source_id existan en el nuevo dataframe.
_selected_source_ids = set()


def _on_pairplot_selection_change(attr, old, new):
    """Sincroniza el conjunto de source_id seleccionados con los índices del CDS."""
    global _selected_source_ids
    ids = pairplot_source.data.get("source_id", [])
    if not ids:
        return
    # Si el usuario vacía la selección, reflejarlo.
    if not new:
        _selected_source_ids = set()
        return
    # Convertir índices -> ids (defensivo ante índices fuera de rango).
    sel_ids = []
    n = len(ids)
    for k in new:
        if 0 <= k < n:
            sel_ids.append(str(ids[k]))
    _selected_source_ids = set(sel_ids)


def _reapply_pairplot_selection():
    """Re-aplica la selección actual (por source_id) a los índices del CDS."""
    if not _selected_source_ids:
        pairplot_source.selected.indices = []
        return
    ids = pairplot_source.data.get("source_id", [])
    if not ids:
        pairplot_source.selected.indices = []
        return
    ids_arr = np.asarray(ids, dtype=str)
    # np.isin es rápido incluso para arrays grandes
    mask = np.isin(ids_arr, list(_selected_source_ids))
    pairplot_source.selected.indices = np.nonzero(mask)[0].tolist()


pairplot_source.selected.on_change("indices", _on_pairplot_selection_change)

# Diagrama color-magnitud: inicialmente vacío o todo el conjunto
cmd_source = ColumnDataSource(data={"x": df["BP_RP"].values, "y": df["MG"].values})

# ---------------------------------------------------------------------------
# Pairplot (500x500, tamaño 0.5, alpha 1)
# ---------------------------------------------------------------------------
pair_plot = figure(
    width=500,
    height=500,
    title="Latent Dimensions (lasso selection)",
    tools="pan,zoom_in,zoom_out,box_zoom,reset,save",
    active_drag="pan",
)
# continuous=False: no calcular selección en cada movimiento, solo al soltar el lazo
pair_plot.add_tools(LassoSelectTool(name="lasso", continuous=False))
# Colores de selección contrastantes: seleccionados naranja fuerte, resto muy tenue
pair_plot.scatter(
    source=pairplot_source,
    x="x",
    y="y",
    size=0.5,
    alpha=1.0,
    color="steelblue",
    selection_fill_color="darkorange",
    selection_line_color="darkorange",
    selection_alpha=1.0,
    nonselection_fill_color="lightgray",
    nonselection_line_color="lightgray",
    nonselection_alpha=0.15,
)
pair_plot.xaxis.axis_label = dim_cols[0]
pair_plot.yaxis.axis_label = dim_cols[1]

# ---------------------------------------------------------------------------
# Diagrama color-magnitud (puntos más visibles)
# ---------------------------------------------------------------------------
cmd_plot = figure(
    width=1300,
    height=650,
    title="CAMD (BP_RP vs MG)",
    tools="pan,zoom_in,zoom_out,box_zoom,reset,save",
    x_axis_label="BP_RP",
    y_axis_label="MG",
    y_range=DataRange1d(flipped=True),
)
cmd_scatter = cmd_plot.scatter(
    source=cmd_source,
    x="x",
    y="y",
    size=0.5,
    alpha=1.0,
)

# ---------------------------------------------------------------------------
# Visualización de espectros (mediana selección vs mediana global; espectros desde reduced_spectra.parquet)
# ---------------------------------------------------------------------------
spectrum_all_bp = ColumnDataSource(data={"x": [], "y": []})
spectrum_all_rp = ColumnDataSource(data={"x": [], "y": []})
spectrum_sel_bp = ColumnDataSource(data={"x": [], "y": []})
spectrum_sel_rp = ColumnDataSource(data={"x": [], "y": []})

def _get_spectra_status_text():
    if _SPECTRA_DF is not None and _global_median_bp is not None:
        n_spec = len(_SPECTRA_DF)
        return f"<span style='color:green'>Espectros cargados desde reduced_spectra.parquet ({n_spec:,} fuentes).</span>"
    if _SPECTRA_DF is not None and _global_median_bp is None:
        return "<span style='color:orange'>Espectros cargados; faltan filtered_wl_bp.npy / filtered_wl_rp.npy en el directorio raíz.</span>"
    return "<span style='color:gray'>Espectros: no se encontró reduced_spectra.parquet en el directorio raíz.</span>"

spectra_status = Div(
    text=_get_spectra_status_text(),
    width=500,
    height=30,
)

# Rellenar mediana global al inicio si ya está cargada
if _global_median_bp is not None and _wl_bp is not None:
    spectrum_all_bp.data = {"x": _wl_bp, "y": _global_median_bp}
if _global_median_rp is not None and _wl_rp is not None:
    spectrum_all_rp.data = {"x": _wl_rp, "y": _global_median_rp}

spec_bp_plot = figure(
    width=500,
    height=280,
    title="Espectro BP (mediana global vs selección)",
    tools="pan,zoom_in,zoom_out,box_zoom,reset,save",
    x_axis_label="Longitud de onda [nm]",
    y_axis_label="Flujo",
)
spec_bp_plot.line("x", "y", source=spectrum_all_bp, line_width=1.5, color="gray", alpha=0.8, legend_label="Todos")
spec_bp_plot.line("x", "y", source=spectrum_sel_bp, line_width=2, color="darkorange", legend_label="Selección")

spec_rp_plot = figure(
    width=500,
    height=280,
    title="Espectro RP (mediana global vs selección)",
    tools="pan,zoom_in,zoom_out,box_zoom,reset,save",
    x_axis_label="Longitud de onda [nm]",
    y_axis_label="Flujo",
)
spec_rp_plot.line("x", "y", source=spectrum_all_rp, line_width=1.5, color="gray", alpha=0.8, legend_label="Todos")
spec_rp_plot.line("x", "y", source=spectrum_sel_rp, line_width=2, color="darkorange", legend_label="Selección")


# ---------------------------------------------------------------------------
# Espectros por bins de color y magnitud (todas las fuentes + selección)
# ---------------------------------------------------------------------------
def _init_bin_spectra_all():
    """Inicializa bins de color y magnitud y calcula las medianas de espectros
    para todas las fuentes en cada bin. También crea las figuras por bin.
    Se inspira en spectra_plots.py pero adaptado a Bokeh.
    """
    global _bin_mg_intervals, _bin_color_intervals
    global _bp_all_bin_sources, _bp_sel_bin_sources, _rp_all_bin_sources, _rp_sel_bin_sources
    global _bp_bin_figs, _rp_bin_figs

    # Reinciar estructuras
    _bp_all_bin_sources = {}
    _bp_sel_bin_sources = {}
    _rp_all_bin_sources = {}
    _rp_sel_bin_sources = {}
    _bp_bin_figs = []
    _rp_bin_figs = []
    _bin_mg_intervals = None
    _bin_color_intervals = None

    if _SPECTRA_DF is None or _wl_bp is None or _wl_rp is None:
        return

    # DataFrame base para binificar: si ya hay df de dimensiones cargado úsalo
    # (para que los rangos coincidan con el CAMD), si no, intenta usar el
    # propio parquet de espectros si contiene BP_RP y MG.
    if not df.empty and {"BP_RP", "MG", "source_id"}.issubset(df.columns):
        bin_df = df[["source_id", "BP_RP", "MG"]].copy()
    elif {"BP_RP", "MG", "source_id"}.issubset(_SPECTRA_DF.reset_index().columns):
        tmp_all = _SPECTRA_DF.reset_index()
        bin_df = tmp_all[["source_id", "BP_RP", "MG"]].copy()
    else:
        # No hay información de color/magnitud asociada a los espectros.
        return

    # Definir bordes de bins siguiendo la idea de spectra_plots.py
    n_bins = _BINS_CAMD
    color_edges = np.linspace(bin_df["BP_RP"].min(), bin_df["BP_RP"].max(), n_bins + 1)
    mg_edges = np.linspace(bin_df["MG"].min(), bin_df["MG"].max(), n_bins + 1)

    tmp = bin_df.copy()
    tmp["color_bin"] = pd.cut(tmp["BP_RP"], bins=color_edges, include_lowest=True)
    tmp["mg_bin"] = pd.cut(tmp["MG"], bins=mg_edges, include_lowest=True)

    if tmp["mg_bin"].isna().all() or tmp["color_bin"].isna().all():
        return

    tmp = tmp.dropna(subset=["mg_bin", "color_bin"])
    if tmp.empty:
        return

    _bin_mg_intervals = list(tmp["mg_bin"].cat.categories)
    _bin_color_intervals = list(tmp["color_bin"].cat.categories)

    # Paleta sencilla por bin de color (como aproximación a cmap continuo)
    BIN_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    # Crear figuras por bin de magnitud para BP y RP
    for i, mg_bin in enumerate(_bin_mg_intervals):
        fig_bp = figure(
            width=500,
            height=280,
            title=f"BP – MG ∈ {mg_bin}",
            tools="pan,zoom_in,zoom_out,box_zoom,reset,save",
            x_axis_label="Longitud de onda [nm]" if i == len(_bin_mg_intervals) - 1 else "",
            y_axis_label="Flujo",
        )
        fig_rp = figure(
            width=500,
            height=280,
            title=f"RP – MG ∈ {mg_bin}",
            tools="pan,zoom_in,zoom_out,box_zoom,reset,save",
            x_axis_label="Longitud de onda [nm]" if i == len(_bin_mg_intervals) - 1 else "",
            y_axis_label="Flujo",
        )

        for j, color_bin in enumerate(_bin_color_intervals):
            key = (i, j)
            # Fuentes en este bin
            mask = (tmp["mg_bin"] == mg_bin) & (tmp["color_bin"] == color_bin)
            sids = tmp.loc[mask, "source_id"].values

            if len(sids) > 0:
                med_bp, med_rp = _get_spectra_for_source_ids(sids)
            else:
                med_bp, med_rp = None, None

            if med_bp is None:
                y_bp_all = np.full_like(_wl_bp, np.nan, dtype=float)
            else:
                y_bp_all = med_bp.astype(float)
            if med_rp is None:
                y_rp_all = np.full_like(_wl_rp, np.nan, dtype=float)
            else:
                y_rp_all = med_rp.astype(float)

            bp_all_src = ColumnDataSource(data={"x": _wl_bp, "y": y_bp_all})
            bp_sel_src = ColumnDataSource(data={"x": _wl_bp, "y": np.full_like(_wl_bp, np.nan, dtype=float)})
            rp_all_src = ColumnDataSource(data={"x": _wl_rp, "y": y_rp_all})
            rp_sel_src = ColumnDataSource(data={"x": _wl_rp, "y": np.full_like(_wl_rp, np.nan, dtype=float)})

            _bp_all_bin_sources[key] = bp_all_src
            _bp_sel_bin_sources[key] = bp_sel_src
            _rp_all_bin_sources[key] = rp_all_src
            _rp_sel_bin_sources[key] = rp_sel_src

            color = BIN_COLORS[j % len(BIN_COLORS)]

            # Todas las fuentes en el bin
            fig_bp.line("x", "y", source=bp_all_src, line_width=1.0, color=color, alpha=0.9, legend_label=str(color_bin))
            fig_rp.line("x", "y", source=rp_all_src, line_width=1.0, color=color, alpha=0.9, legend_label=str(color_bin))

            # Mediana de la selección (se rellenará al pulsar el botón)
            fig_bp.line("x", "y", source=bp_sel_src, line_width=2.0, color=color, alpha=1.0, line_dash="dashed")
            fig_rp.line("x", "y", source=rp_sel_src, line_width=2.0, color=color, alpha=1.0, line_dash="dashed")

        fig_bp.legend.click_policy = "hide"
        fig_rp.legend.click_policy = "hide"

        _bp_bin_figs.append(fig_bp)
        _rp_bin_figs.append(fig_rp)

    # Actualizar las columnas del layout con las nuevas figuras
    bp_bins_column.children = _bp_bin_figs
    rp_bins_column.children = _rp_bin_figs


def _init_bin_spectra_from_precomputed():
    """Inicializa la vista por bins usando medianas precalculadas (sin recalcular espectros)."""
    global _bin_mg_intervals, _bin_color_intervals
    global _bp_all_bin_sources, _bp_sel_bin_sources, _rp_all_bin_sources, _rp_sel_bin_sources
    global _bp_bin_figs, _rp_bin_figs
    global _precomputed_mg_edges, _precomputed_color_edges

    # Reiniciar estructuras
    _bp_all_bin_sources = {}
    _bp_sel_bin_sources = {}
    _rp_all_bin_sources = {}
    _rp_sel_bin_sources = {}
    _bp_bin_figs = []
    _rp_bin_figs = []
    _bin_mg_intervals = None
    _bin_color_intervals = None

    if _wl_bp is None or _wl_rp is None:
        return

    mg_edges, color_edges, med_dict = load_precomputed_bin_medians()
    if mg_edges is None or color_edges is None or not med_dict:
        return

    _precomputed_mg_edges = mg_edges
    _precomputed_color_edges = color_edges

    # Intervalos por bins a partir de bordes precalculados
    _bin_mg_intervals = list(pd.IntervalIndex.from_breaks(mg_edges))
    _bin_color_intervals = list(pd.IntervalIndex.from_breaks(color_edges))

    BIN_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    for i, mg_bin in enumerate(_bin_mg_intervals):
        fig_bp = figure(
            width=500,
            height=280,
            title=f"BP – MG ∈ {mg_bin}",
            tools="pan,zoom_in,zoom_out,box_zoom,reset,save",
            x_axis_label="Longitud de onda [nm]" if i == len(_bin_mg_intervals) - 1 else "",
            y_axis_label="Flujo",
        )
        fig_rp = figure(
            width=500,
            height=280,
            title=f"RP – MG ∈ {mg_bin}",
            tools="pan,zoom_in,zoom_out,box_zoom,reset,save",
            x_axis_label="Longitud de onda [nm]" if i == len(_bin_mg_intervals) - 1 else "",
            y_axis_label="Flujo",
        )

        for j, color_bin in enumerate(_bin_color_intervals):
            key = (i, j)
            dict_key = (float(mg_bin.left), float(mg_bin.right), float(color_bin.left), float(color_bin.right))
            bp_med, rp_med, _n = med_dict.get(dict_key, (None, None, 0))

            if bp_med is None:
                y_bp_all = np.full_like(_wl_bp, np.nan, dtype=float)
            else:
                y_bp_all = bp_med.astype(float)
                if len(y_bp_all) != len(_wl_bp):
                    y_bp_all = y_bp_all[: len(_wl_bp)] if len(y_bp_all) >= len(_wl_bp) else np.pad(
                        y_bp_all.astype(float), (0, len(_wl_bp) - len(y_bp_all)), "constant", constant_values=np.nan
                    )

            if rp_med is None:
                y_rp_all = np.full_like(_wl_rp, np.nan, dtype=float)
            else:
                y_rp_all = rp_med.astype(float)
                if len(y_rp_all) != len(_wl_rp):
                    y_rp_all = y_rp_all[: len(_wl_rp)] if len(y_rp_all) >= len(_wl_rp) else np.pad(
                        y_rp_all.astype(float), (0, len(_wl_rp) - len(y_rp_all)), "constant", constant_values=np.nan
                    )

            bp_all_src = ColumnDataSource(data={"x": _wl_bp, "y": y_bp_all})
            bp_sel_src = ColumnDataSource(data={"x": _wl_bp, "y": np.full_like(_wl_bp, np.nan, dtype=float)})
            rp_all_src = ColumnDataSource(data={"x": _wl_rp, "y": y_rp_all})
            rp_sel_src = ColumnDataSource(data={"x": _wl_rp, "y": np.full_like(_wl_rp, np.nan, dtype=float)})

            _bp_all_bin_sources[key] = bp_all_src
            _bp_sel_bin_sources[key] = bp_sel_src
            _rp_all_bin_sources[key] = rp_all_src
            _rp_sel_bin_sources[key] = rp_sel_src

            color = BIN_COLORS[j % len(BIN_COLORS)]
            fig_bp.line("x", "y", source=bp_all_src, line_width=1.0, color=color, alpha=0.9, legend_label=str(color_bin))
            fig_rp.line("x", "y", source=rp_all_src, line_width=1.0, color=color, alpha=0.9, legend_label=str(color_bin))
            fig_bp.line("x", "y", source=bp_sel_src, line_width=2.0, color=color, alpha=1.0, line_dash="dashed")
            fig_rp.line("x", "y", source=rp_sel_src, line_width=2.0, color=color, alpha=1.0, line_dash="dashed")

        fig_bp.legend.click_policy = "hide"
        fig_rp.legend.click_policy = "hide"
        _bp_bin_figs.append(fig_bp)
        _rp_bin_figs.append(fig_rp)

    bp_bins_column.children = _bp_bin_figs
    rp_bins_column.children = _rp_bin_figs

# ---------------------------------------------------------------------------
# Controles
# ---------------------------------------------------------------------------
status_div = Div(text="", width=500, height=30)
path_input = TextInput(
    title="Ruta al parquet de dimensiones (relativa al directorio de la app)",
    placeholder="ej. dimensions.parquet o carpeta/dimensions.parquet",
    width=450,
)

select_dim1 = Select(title="Dimension 1", value=dim_cols[0], options=dim_cols)
select_dim2 = Select(title="Dimension 2", value=dim_cols[1], options=dim_cols)

CMD_SIZE_OPTIONS = ["0.05", "0.5", "1"]
select_cmd_size = Select(
    title="Tamaño puntos CAMD",
    value="0.5",
    options=CMD_SIZE_OPTIONS,
    width=150,
)


def _on_cmd_size_change(attr, old, new):
    if new and cmd_scatter.glyph:
        cmd_scatter.glyph.size = float(new)


select_cmd_size.on_change("value", _on_cmd_size_change)


def _apply_loaded_df(new_df: pd.DataFrame) -> bool:
    """Actualiza datos y desplegables con el parquet de dimensiones. No rellena pairplot ni CAMD (evita ralentizar)."""
    global df, dim_cols
    df = new_df
    dim_cols = sorted([c for c in df.columns if is_dim_column(c)])
    if not dim_cols:
        dim_cols = sorted([c for c in df.columns if c not in ("source_id", "BP_RP", "MG")])
    if len(dim_cols) < 2:
        status_div.text = "<span style='color:orange'>Se requieren al menos 2 dimensiones.</span>"
        return False
    status_div.text = f"<span style='color:green'>Cargado: {len(df):,} fuentes. Usa «Show Pairplot» y «Update CAMD» para dibujar.</span>"
    select_dim1.options = dim_cols
    select_dim2.options = dim_cols
    select_dim1.value = dim_cols[0]
    select_dim2.value = dim_cols[1]
    # No enviar datos al pairplot ni CAMD aquí (ralentiza); se rellenan con «Show Pairplot» / «Update CAMD»
    pair_plot.xaxis.axis_label = dim_cols[0]
    pair_plot.yaxis.axis_label = dim_cols[1]
    # Importante: no borramos _selected_source_ids; la selección se mantiene por source_id.
    # Pero al vaciar el CDS, limpiamos los índices para evitar incoherencias visuales.
    pairplot_source.selected.indices = []
    pairplot_source.data = {"x": [], "y": [], "source_id": []}
    cmd_source.data = {"x": [], "y": []}
    info.text = f"Sources: {len(df):,}"
    # Importante para rendimiento: cargar dimensiones NO debe recalcular espectros ni medianas por bins.
    # La inicialización de espectros por bins se hace al arrancar (si es posible) o cuando se necesite.

    return True


def load_from_path():
    raw = (path_input.value or "").strip()
    if not raw:
        status_div.text = "<span style='color:orange'>Indica la ruta al parquet de dimensiones.</span>"
        return
    # Solo ruta relativa al directorio donde está app.py
    p = (Path(__file__).resolve().parent / raw).resolve()
    if not p.is_file():
        status_div.text = f"<span style='color:red'>El archivo no existe: {p}</span>"
        return
    try:
        new_df = load_data_from_path(str(p))
        ok = _apply_loaded_df(new_df)
        if not ok:
            # _apply_loaded_df ya habrá actualizado status_div con el motivo.
            return
        # Mensaje de éxito explícito para el usuario.
        status_div.text = f"<span style='color:green'>Cargado correctamente: {len(new_df):,} fuentes.</span>"
    except Exception as e:
        status_div.text = f"<span style='color:red'>Error al cargar: {e}</span>"


btn_load_path = Button(label="Load from path", button_type="default")
btn_load_path.on_click(load_from_path)


def show_pairplot():
    d1 = select_dim1.value
    d2 = select_dim2.value
    pairplot_source.data = {
        "x": df[d1].values,
        "y": df[d2].values,
        "source_id": df["source_id"].astype(str).values,
    }
    pair_plot.xaxis.axis_label = d1
    pair_plot.yaxis.axis_label = d2
    # Reaplicar selección persistida (por source_id) en el nuevo CDS.
    _reapply_pairplot_selection()


def _get_spectra_for_source_ids(source_ids):
    """Obtiene arrays BP y RP para una lista de source_id desde _SPECTRA_DF (en memoria).
    Normaliza todas las longitudes a n_bp/n_rp para poder hacer stack aunque haya espectros de distinta longitud."""
    if _SPECTRA_DF is None or _wl_bp is None or _wl_rp is None:
        return None, None
    n_bp, n_rp = len(_wl_bp), len(_wl_rp)
    list_bp = []
    list_rp = []
    for sid in source_ids:
        try:
            row = _SPECTRA_DF.loc[sid]
            bp_arr = _to_array(row["BP"])
            rp_arr = _to_array(row["RP"])
            if _bp_sort_idx is not None and len(bp_arr) == len(_bp_sort_idx):
                bp_arr = bp_arr[_bp_sort_idx]
            # Unificar longitud: truncar o rellenar con nan para que np.stack no falle
            if len(bp_arr) >= n_bp:
                bp_arr = bp_arr[:n_bp].astype(float)
            else:
                bp_arr = np.pad(bp_arr.astype(float), (0, n_bp - len(bp_arr)), "constant", constant_values=np.nan)
            if len(rp_arr) >= n_rp:
                rp_arr = rp_arr[:n_rp].astype(float)
            else:
                rp_arr = np.pad(rp_arr.astype(float), (0, n_rp - len(rp_arr)), "constant", constant_values=np.nan)
            list_bp.append(bp_arr)
            list_rp.append(rp_arr)
        except (KeyError, TypeError):
            continue
    if not list_bp:
        return None, None
    med_bp = np.nanmedian(np.stack(list_bp), axis=0)
    med_rp = np.nanmedian(np.stack(list_rp), axis=0)
    return med_bp, med_rp


# Div para barra de progreso / tiempo de actualización CAMD y espectros
PROGRESS_HTML_IDLE = "<span style='color:#888'>—</span>"
PROGRESS_HTML_LOADING = """
<span style='color:#06c'>Cargando CAMD y espectros…</span>
<div style='width:220px;height:10px;background:#e0e0e0;border-radius:5px;overflow:hidden;margin-top:6px'>
  <div style='height:100%;width:60%;background:#2196F3;border-radius:5px'></div>
</div>
"""

progress_div = Div(
    text=PROGRESS_HTML_IDLE,
    width=500,
    height=50,
    styles={"min-height": "50px"},
)


def _do_update_diagram():
    """Realiza la actualización del CAMD y espectros y actualiza el div de progreso con el tiempo."""
    global _cached_sel_median_bp, _cached_sel_median_rp, _cached_sel_bin_bp, _cached_sel_bin_rp
    t0 = time.perf_counter()
    sel = pairplot_source.selected.indices
    if not sel:
        # Sin selección: CAMD completo y espectros de referencia = medianas globales.
        cmd_source.data = {"x": df["BP_RP"].values, "y": df["MG"].values}
        if _global_median_bp is not None and _wl_bp is not None:
            spectrum_sel_bp.data = {"x": _wl_bp, "y": _global_median_bp}
        if _global_median_rp is not None and _wl_rp is not None:
            spectrum_sel_rp.data = {"x": _wl_rp, "y": _global_median_rp}

        # En la vista por bins solo se muestran las curvas de "todas las fuentes".
        for key, src in _bp_sel_bin_sources.items():
            src.data = {"x": _wl_bp, "y": np.full_like(_wl_bp, np.nan, dtype=float)}
        for key, src in _rp_sel_bin_sources.items():
            src.data = {"x": _wl_rp, "y": np.full_like(_wl_rp, np.nan, dtype=float)}
        _cached_sel_median_bp = None
        _cached_sel_median_rp = None
        _cached_sel_bin_bp = {}
        _cached_sel_bin_rp = {}
    else:
        # Con selección: CAMD restringido y medianas de los espectros seleccionados.
        sub = df.iloc[sel]
        cmd_source.data = {"x": sub["BP_RP"].values, "y": sub["MG"].values}
        source_ids = sub["source_id"].values
        med_bp, med_rp = _get_spectra_for_source_ids(source_ids)
        _cached_sel_median_bp = med_bp.astype(float) if med_bp is not None else None
        _cached_sel_median_rp = med_rp.astype(float) if med_rp is not None else None
        if med_bp is not None and med_rp is not None:
            spectrum_sel_bp.data = {"x": _wl_bp, "y": med_bp}
            spectrum_sel_rp.data = {"x": _wl_rp, "y": med_rp}

        # Actualizar medianas de la selección por bin de color y magnitud.
        _cached_sel_bin_bp = {}
        _cached_sel_bin_rp = {}
        if _bin_mg_intervals is not None and _bin_color_intervals is not None:
            n_bins = len(_bin_mg_intervals)
            if _precomputed_color_edges is not None and _precomputed_mg_edges is not None:
                color_edges = _precomputed_color_edges
                mg_edges = _precomputed_mg_edges
            else:
                color_edges = np.linspace(df["BP_RP"].min(), df["BP_RP"].max(), n_bins + 1)
                mg_edges = np.linspace(df["MG"].min(), df["MG"].max(), n_bins + 1)

            tmp_sel = sub[["source_id", "BP_RP", "MG"]].copy()
            # Cambio mínimo: evitar comparar objetos Interval (problemático en el primer bin con include_lowest=True).
            # En su lugar, calculamos índices (i,j) directamente con los bordes.
            tmp_sel["i_mg"] = pd.cut(tmp_sel["MG"], bins=mg_edges, include_lowest=True, labels=False)
            tmp_sel["j_color"] = pd.cut(tmp_sel["BP_RP"], bins=color_edges, include_lowest=True, labels=False)
            tmp_sel = tmp_sel.dropna(subset=["i_mg", "j_color"])
            tmp_sel["i_mg"] = tmp_sel["i_mg"].astype(int)
            tmp_sel["j_color"] = tmp_sel["j_color"].astype(int)

            for i in range(len(_bin_mg_intervals)):
                for j in range(len(_bin_color_intervals)):
                    key = (i, j)
                    if key not in _bp_sel_bin_sources or key not in _rp_sel_bin_sources:
                        continue
                    mask = (tmp_sel["i_mg"] == i) & (tmp_sel["j_color"] == j)
                    sids_bin = tmp_sel.loc[mask, "source_id"].values
                    if len(sids_bin) > 0:
                        mb, mr = _get_spectra_for_source_ids(sids_bin)
                    else:
                        mb, mr = None, None

                    if mb is None:
                        y_bp_sel = np.full_like(_wl_bp, np.nan, dtype=float)
                    else:
                        y_bp_sel = mb.astype(float)
                    if mr is None:
                        y_rp_sel = np.full_like(_wl_rp, np.nan, dtype=float)
                    else:
                        y_rp_sel = mr.astype(float)

                    _cached_sel_bin_bp[key] = y_bp_sel
                    _cached_sel_bin_rp[key] = y_rp_sel

            # Aplicar datos según el modo seleccionado, reutilizando lo precalculado.
            if _selection_medians_mode == "global" and _cached_sel_median_bp is not None and _cached_sel_median_rp is not None:
                for key, src in _bp_sel_bin_sources.items():
                    src.data = {"x": _wl_bp, "y": _cached_sel_median_bp}
                for key, src in _rp_sel_bin_sources.items():
                    src.data = {"x": _wl_rp, "y": _cached_sel_median_rp}
            else:
                for key, src in _bp_sel_bin_sources.items():
                    y_bp = _cached_sel_bin_bp.get(key, np.full_like(_wl_bp, np.nan, dtype=float))
                    src.data = {"x": _wl_bp, "y": y_bp}
                for key, src in _rp_sel_bin_sources.items():
                    y_rp = _cached_sel_bin_rp.get(key, np.full_like(_wl_rp, np.nan, dtype=float))
                    src.data = {"x": _wl_rp, "y": y_rp}
    elapsed = time.perf_counter() - t0
    progress_div.text = f"<span style='color:green'>Listo en {elapsed:.2f} s</span>"


def update_diagram():
    """Programa la actualización en el siguiente tick para que se muestre 'Cargando…' antes del trabajo."""
    progress_div.text = PROGRESS_HTML_LOADING
    curdoc().add_next_tick_callback(_do_update_diagram)


btn_show = Button(label="Show Pairplot", button_type="primary")
btn_show.on_click(show_pairplot)

btn_update_cmd = Button(label="Update CAMD & waveforms", button_type="success")
btn_update_cmd.on_click(update_diagram)


def switch_medians_mode():
    """Alterna entre medianas por bin y mediana global de selección (negra) en bins."""
    global _selection_medians_mode
    if _selection_medians_mode == "by_bin":
        _selection_medians_mode = "global"
        for fig in _bp_bin_figs + _rp_bin_figs:
            if len(fig.renderers) > 0:
                for r in fig.renderers:
                    if hasattr(r, "glyph") and getattr(r.glyph, "line_dash", None) == "dashed":
                        r.glyph.line_color = "black"
    else:
        _selection_medians_mode = "by_bin"
        # Restaurar color por bin en líneas de selección siguiendo el orden de bins de color.
        BIN_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
        for fig in _bp_bin_figs + _rp_bin_figs:
            dashed_idx = 0
            for r in fig.renderers:
                if hasattr(r, "glyph") and getattr(r.glyph, "line_dash", None) == "dashed":
                    r.glyph.line_color = BIN_COLORS[dashed_idx % len(BIN_COLORS)]
                    dashed_idx += 1

    # Aplicar inmediatamente el modo actual sin recalcular espectros.
    if _wl_bp is None or _wl_rp is None:
        return
    if _selection_medians_mode == "global" and _cached_sel_median_bp is not None and _cached_sel_median_rp is not None:
        for key, src in _bp_sel_bin_sources.items():
            src.data = {"x": _wl_bp, "y": _cached_sel_median_bp}
        for key, src in _rp_sel_bin_sources.items():
            src.data = {"x": _wl_rp, "y": _cached_sel_median_rp}
    else:
        for key, src in _bp_sel_bin_sources.items():
            y_bp = _cached_sel_bin_bp.get(key, np.full_like(_wl_bp, np.nan, dtype=float))
            src.data = {"x": _wl_bp, "y": y_bp}
        for key, src in _rp_sel_bin_sources.items():
            y_rp = _cached_sel_bin_rp.get(key, np.full_like(_wl_rp, np.nan, dtype=float))
            src.data = {"x": _wl_rp, "y": y_rp}


btn_switch_medians = Button(label="switch medians mode", button_type="default")
btn_switch_medians.on_click(switch_medians_mode)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
download_btn = Button(
    label="Download source_id (selection)",
    button_type="warning",
)
download_btn.js_on_click(CustomJS(
    args=dict(source=pairplot_source),
    code="""
    const indices = source.selected.indices;
    const ids = source.data.source_id;

    if (indices.length === 0) {
        alert("No selection");
        return;
    }

    let text = "";
    for (let i = 0; i < indices.length; i++) {
        const k = indices[i];
        text += ids[k].toString() + "\\n";
    }

    const blob = new Blob([text], { type: "text/plain;charset=utf-8;" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "selected_source_ids";
    a.click();

    URL.revokeObjectURL(url);
    """
))


info = Paragraph(
    text=f"Sources: {len(df):,}",
    width=500,
)
controls = column(
    row(path_input, btn_load_path),
    status_div,
    info,
    row(select_dim1, select_dim2),
    btn_show,
    pair_plot,
    row(btn_update_cmd, btn_switch_medians, download_btn, select_cmd_size),
    progress_div,
)

# Layout principal:
# - Arriba: controles + pairplot a la izquierda; espectros a la derecha.
# - Abajo: CAMD debajo de todo.
spectra_panel = column(
    spectra_status,
    row(spec_bp_plot, spec_rp_plot),
    row(bp_bins_column, rp_bins_column),
)

layout = column(
    row(controls, spectra_panel),
    cmd_plot,
)

curdoc().add_root(layout)
curdoc().title = "Gap Seeker"

# Nota: no inicializamos automáticamente las medianas por bins al arrancar.
# Es caro (recorre muchos espectros) y, si existen medianas precalculadas,
# no tiene sentido recomputarlas aquí.

# Si existen medianas precalculadas por bins, inicializa la vista de bins sin recomputar espectros.
try:
    _init_bin_spectra_from_precomputed()
except Exception:
    pass
