"""
Precalcula medianas de espectros (global y por bins de MG/BP_RP) y las guarda en Parquet.

Uso (PowerShell):
  python .\precompute_spectra_medians.py --spectra .\reduced_spectra.parquet --out .\spectra_medians.parquet
  python .\precompute_spectra_medians.py --spectra .\reduced_spectra.parquet --dims .\dimensions.parquet --out .\spectra_medians.parquet

Notas:
- Replica la lógica de longitudes de onda y ordenado BP de `app_v2.py` (filtered_wl_*.npy y argsort).
- Replica la binificación de `app_v2.py`: bins uniformes (linspace) entre min y max del dataset.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent


def _to_array(v) -> np.ndarray:
    if isinstance(v, str):
        v = ast.literal_eval(v)
    if isinstance(v, (list, np.ndarray)) and len(v) > 0:
        return np.array(v, dtype=float)
    return np.array([np.nan], dtype=float)


def load_wavelengths(root_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    wl_bp_path = root_dir / "filtered_wl_bp.npy"
    wl_rp_path = root_dir / "filtered_wl_rp.npy"
    if not wl_bp_path.is_file():
        wl_bp_path = root_dir / "wl_bp.npy"
    if not wl_rp_path.is_file():
        wl_rp_path = root_dir / "wl_rp.npy"
    wl_bp = np.load(wl_bp_path) if wl_bp_path.is_file() else None
    wl_rp = np.load(wl_rp_path) if wl_rp_path.is_file() else None
    bp_sort_idx = None
    if wl_bp is not None:
        bp_sort_idx = np.argsort(wl_bp)
        wl_bp = wl_bp[bp_sort_idx]
    return wl_bp, wl_rp, bp_sort_idx


def _normalize_bp_rp_lengths(
    bp_arr: np.ndarray,
    rp_arr: np.ndarray,
    n_bp: int,
    n_rp: int,
    bp_sort_idx: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if bp_sort_idx is not None and len(bp_arr) == len(bp_sort_idx):
        bp_arr = bp_arr[bp_sort_idx]
    if len(bp_arr) >= n_bp:
        bp_arr = bp_arr[:n_bp].astype(float)
    else:
        bp_arr = np.pad(bp_arr.astype(float), (0, n_bp - len(bp_arr)), "constant", constant_values=np.nan)
    if len(rp_arr) >= n_rp:
        rp_arr = rp_arr[:n_rp].astype(float)
    else:
        rp_arr = np.pad(rp_arr.astype(float), (0, n_rp - len(rp_arr)), "constant", constant_values=np.nan)
    return bp_arr, rp_arr


def _median_of_source_ids(
    spec_df: pd.DataFrame,
    source_ids: Iterable,
    n_bp: int,
    n_rp: int,
    bp_sort_idx: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    list_bp = []
    list_rp = []
    n_used = 0
    for sid in source_ids:
        try:
            row = spec_df.loc[sid]
        except KeyError:
            continue
        bp_arr = _to_array(row["BP"])
        rp_arr = _to_array(row["RP"])
        bp_arr, rp_arr = _normalize_bp_rp_lengths(bp_arr, rp_arr, n_bp=n_bp, n_rp=n_rp, bp_sort_idx=bp_sort_idx)
        list_bp.append(bp_arr)
        list_rp.append(rp_arr)
        n_used += 1
    if n_used == 0:
        return None, None, 0
    med_bp = np.nanmedian(np.stack(list_bp), axis=0)
    med_rp = np.nanmedian(np.stack(list_rp), axis=0)
    return med_bp, med_rp, n_used


def _bin_edges(series: pd.Series, n_bins: int) -> np.ndarray:
    # Misma filosofía que `app_v2.py`: linspace(min, max, n_bins + 1)
    vmin = float(np.nanmin(series.values))
    vmax = float(np.nanmax(series.values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        # Evita bins degenerados: crea un rango mínimo.
        vmax = vmin + 1.0
    return np.linspace(vmin, vmax, n_bins + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra", type=str, default=str(ROOT_DIR / "reduced_spectra.parquet"))
    ap.add_argument("--dims", type=str, default="")
    ap.add_argument("--out", type=str, default=str(ROOT_DIR / "spectra_medians.parquet"))
    ap.add_argument("--bins", type=int, default=4)
    args = ap.parse_args()

    spectra_path = Path(args.spectra).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    dims_path = Path(args.dims).expanduser().resolve() if args.dims else None

    if not spectra_path.is_file():
        raise FileNotFoundError(f"No existe el parquet de espectros: {spectra_path}")

    wl_bp, wl_rp, bp_sort_idx = load_wavelengths(ROOT_DIR)
    if wl_bp is None or wl_rp is None:
        raise FileNotFoundError(
            "No se encontraron ejes de longitudes de onda (filtered_wl_bp.npy/filtered_wl_rp.npy o wl_bp.npy/wl_rp.npy) "
            f"en {ROOT_DIR}"
        )

    n_bp, n_rp = len(wl_bp), len(wl_rp)

    spec_df = pd.read_parquet(spectra_path)
    required = {"source_id", "BP", "RP"}
    missing = required - set(spec_df.columns)
    if missing:
        raise ValueError(f"El parquet de espectros debe contener {required}. Faltan: {missing}")
    spec_df = spec_df.set_index("source_id")

    # DataFrame para bins (BP_RP, MG).
    bin_df = None
    if dims_path is not None:
        if not dims_path.is_file():
            raise FileNotFoundError(f"No existe el parquet de dimensiones: {dims_path}")
        dims_df = pd.read_parquet(dims_path)
        needed = {"source_id", "BP_RP", "MG"}
        missing2 = needed - set(dims_df.columns)
        if missing2:
            raise ValueError(f"El parquet de dimensiones debe contener {needed}. Faltan: {missing2}")
        bin_df = dims_df[["source_id", "BP_RP", "MG"]].copy()
    elif {"source_id", "BP_RP", "MG"}.issubset(spec_df.reset_index().columns):
        tmp = spec_df.reset_index()
        bin_df = tmp[["source_id", "BP_RP", "MG"]].copy()

    # Mediana global (sobre todas las fuentes con espectro).
    med_bp_global, med_rp_global, n_global = _median_of_source_ids(
        spec_df, spec_df.index.values, n_bp=n_bp, n_rp=n_rp, bp_sort_idx=bp_sort_idx
    )
    if med_bp_global is None or med_rp_global is None:
        raise RuntimeError("No se pudo calcular la mediana global (¿parquet vacío o sin espectros válidos?).")

    rows = []
    rows.append(
        {
            "kind": "global",
            "n_sources": int(n_global),
            "bins": int(args.bins),
            "mg_left": np.nan,
            "mg_right": np.nan,
            "color_left": np.nan,
            "color_right": np.nan,
            "mg_bin": None,
            "color_bin": None,
            "bp_median": med_bp_global.astype(float).tolist(),
            "rp_median": med_rp_global.astype(float).tolist(),
        }
    )

    # Medianas por bins (si tenemos BP_RP/MG).
    if bin_df is not None and not bin_df.empty:
        n_bins = int(args.bins)
        color_edges = _bin_edges(bin_df["BP_RP"], n_bins=n_bins)
        mg_edges = _bin_edges(bin_df["MG"], n_bins=n_bins)

        tmp = bin_df.copy()
        tmp["color_bin"] = pd.cut(tmp["BP_RP"], bins=color_edges, include_lowest=True)
        tmp["mg_bin"] = pd.cut(tmp["MG"], bins=mg_edges, include_lowest=True)
        tmp = tmp.dropna(subset=["color_bin", "mg_bin"])

        if not tmp.empty:
            mg_cats = list(tmp["mg_bin"].cat.categories)
            color_cats = list(tmp["color_bin"].cat.categories)

            for mg_bin in mg_cats:
                for color_bin in color_cats:
                    mask = (tmp["mg_bin"] == mg_bin) & (tmp["color_bin"] == color_bin)
                    sids = tmp.loc[mask, "source_id"].values
                    med_bp, med_rp, n_used = _median_of_source_ids(
                        spec_df, sids, n_bp=n_bp, n_rp=n_rp, bp_sort_idx=bp_sort_idx
                    )
                    if med_bp is None or med_rp is None:
                        med_bp_list = [float("nan")] * n_bp
                        med_rp_list = [float("nan")] * n_rp
                    else:
                        med_bp_list = med_bp.astype(float).tolist()
                        med_rp_list = med_rp.astype(float).tolist()
                    rows.append(
                        {
                            "kind": "bin",
                            "n_sources": int(n_used),
                            "bins": int(n_bins),
                            "mg_left": float(mg_bin.left),
                            "mg_right": float(mg_bin.right),
                            "color_left": float(color_bin.left),
                            "color_right": float(color_bin.right),
                            "mg_bin": str(mg_bin),
                            "color_bin": str(color_bin),
                            "bp_median": med_bp_list,
                            "rp_median": med_rp_list,
                        }
                    )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print(f"OK: medianas guardadas en {out_path} (filas: {len(out_df):,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

