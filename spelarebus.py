import pandas as pd
import os
import sys
from tqdm import tqdm
import re
import ast
import numpy as np
from gaiaxpy import pwl_to_wl

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns

OUTPUT_PATH = "output_files"
SPECTRA_PATH = "spectra_chunks"
SEED = 42

SPECTRA_FILES_REGEX = r'^output_spectra_\d{3}\.csv$'
SPECTRA_PARQUET_FILENAME = "spectra.parquet"
GAIA_SOURCES_FILENAME = "sources_gaia.csv"
GAIA_SOURCES_PARQUET_FILENAME = "sources.parquet"
MERGED_DATA_PARQUET_FILENAME = "merged_data.parquet"

SPECTRA_SAMPLING = np.linspace(0., 60., 100)
VALID_ALGORITHMS = {"UMAP", "PCA", "AE", "AE_CONV"}
VALID_LATENT_DIMS = {3, 5, 10}

def parse_cli_args():
    if len(sys.argv) != 3:
        raise ValueError(
            "Uso: python spelarebus.py <ALGORITMO> <LATENT_DIM>. "
            "ALGORITMO: UMAP|PCA|AE|AE_CONV. LATENT_DIM: 3|5|10."
        )

    algorithm = sys.argv[1].upper()
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError(
            f"Algoritmo inválido: {algorithm}. "
            "Valores permitidos: UMAP, PCA, AE, AE_CONV."
        )

    try:
        latent_dim = int(sys.argv[2])
    except ValueError as exc:
        raise ValueError(
            f"Latent_dim inválido: {sys.argv[2]}. Debe ser un entero (3, 5 o 10)."
        ) from exc

    if latent_dim not in VALID_LATENT_DIMS:
        raise ValueError(
            f"Latent_dim inválido: {latent_dim}. Valores permitidos: 3, 5, 10."
        )

    return algorithm, latent_dim

def save_pairplots(df, algorithm, latent_dim):
    print("Creando pairplots...")
    bp_cols = [col for col in df.columns if col.startswith('BP_') and col != 'BP_RP']
    rp_cols = [col for col in df.columns if col.startswith('RP_')]

    def filter_outliers(df, cols, lower_q=0.0001, upper_q=0.9999):
        df_filtered = df.copy()
        for col in cols:
            low, high = df[col].quantile(lower_q), df[col].quantile(upper_q)
            df_filtered = df_filtered[df_filtered[col].between(low, high)]
        return df_filtered

    print("Filtrando outliers...")
    df_bp_filtered = filter_outliers(df, bp_cols)
    df_rp_filtered = filter_outliers(df, rp_cols)

    print("Creando plot de componentes pareados para el azul...")
    g_bp = sns.pairplot(
        df_bp_filtered,
        vars=bp_cols,
        diag_kind="kde",
        plot_kws={'s': 3, 'alpha': 0.15, 'color': 'blue'},
        corner=True
    )
    g_bp.fig.suptitle(f'BP {algorithm} Pairplot', y=1.02, fontsize=16)
    for ax in g_bp.axes.flatten():
        if ax is not None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            x_margin = 0.1 * (xmax - xmin)
            y_margin = 0.1 * (ymax - ymin)
            ax.set_xlim(xmin - x_margin, xmax + x_margin)
            ax.set_ylim(ymin - y_margin, ymax + y_margin)
    plt.savefig(f"{OUTPUT_PATH}/bp_{algorithm}_{latent_dim}d_pairplot.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Creando plot de componentes pareados para el rojo...")
    g_rp = sns.pairplot(
        df_rp_filtered,
        vars=rp_cols,
        diag_kind="kde",
        plot_kws={'s': 3, 'alpha': 0.15, 'color': 'red'},
        corner=True
    )
    for ax in g_rp.axes.flatten():
        if ax is not None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            x_margin = 0.1 * (xmax - xmin)
            y_margin = 0.1 * (ymax - ymin)
            ax.set_xlim(xmin - x_margin, xmax + x_margin)
            ax.set_ylim(ymin - y_margin, ymax + y_margin)
    g_rp.fig.suptitle(f'RP {algorithm} Pairplot', y=1.02, fontsize=16)
    plt.savefig(f"{OUTPUT_PATH}/rp_{algorithm}_{latent_dim}d_pairplot.png", dpi=200, bbox_inches="tight")
    plt.close()

def run_umap(latent_dim, df_merged, wl_bp, wl_rp):
    import umap

    bp_len = df_merged["BP"].apply(len)
    rp_len = df_merged["RP"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else -1)

    bp_target = len(wl_bp)
    rp_target = len(wl_rp)

    mask = (bp_len == bp_target) & (rp_len == rp_target)

    df_merged_clean = df_merged[mask].reset_index(drop=True)

    print(f"Objetos válidos: {len(df_merged_clean)}")
    print(f"Objetos descartados: {len(df_merged) - len(df_merged_clean)}")
    
    X_bp = np.vstack(df_merged_clean['BP'].values)
    X_rp = np.vstack(df_merged_clean['RP'].values)

    print("Reducciendo dimensionalidad en azul...")
    reducer_bp = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=latent_dim,
        random_state=SEED,
        n_jobs=-1,
        verbose=True
    )

    print(type(X_bp))
    print(X_bp.shape if hasattr(X_bp, "shape") else "sin shape")
    print(X_bp.dtypes if hasattr(X_bp, "dtypes") else "sin dtypes")

    embedding_bp = reducer_bp.fit_transform(X_bp)

    print("Reducciendo dimensionalidad en rojo...")
    reducer_rp = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=latent_dim,
        random_state=SEED,
        n_jobs=-1,
        verbose=True
    )
    embedding_rp = reducer_rp.fit_transform(X_rp)

    print("Guardando embeddings...")
    umap_columns = {"source_id": df_merged_clean["source_id"].values}
    for i in range(latent_dim):
        dim = i + 1
        umap_columns[f"BP_umap_{dim}"] = embedding_bp[:, i]
        umap_columns[f"RP_umap_{dim}"] = embedding_rp[:, i]

    df_umap = pd.DataFrame(umap_columns)
    df_umap["MG"] = df_merged_clean["MG"].values
    df_umap["BP_RP"] = df_merged_clean["BP_RP"].values

    df_umap.to_parquet(f"{OUTPUT_PATH}/umap_{latent_dim}d_latent.parquet", index=False)
    print(f"Archivo 'umap_{latent_dim}d_latent.parquet' generado con éxito.")
    save_pairplots(df_umap, "UMAP", latent_dim)

def run_pca(latent_dim):
    """Stub para futura implementación de PCA."""
    print(f"[Stub] PCA pendiente de implementar (latent_dim={latent_dim})")

def run_ae(latent_dim):
    """Stub para futura implementación de Autoencoder denso."""
    print(f"[Stub] AE pendiente de implementar (latent_dim={latent_dim})")

def run_ae_conv(latent_dim, df_merged, wl_bp, wl_rp):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando device:", device)

    bp_target = len(wl_bp)
    rp_target = len(wl_rp)
    bp_len = df_merged["BP"].apply(len)
    rp_len = df_merged["RP"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else -1)
    mask = (bp_len == bp_target) & (rp_len == rp_target)
    df_merged_clean = df_merged[mask].reset_index(drop=True)

    print(f"Objetos válidos: {len(df_merged_clean)}")
    print(f"Objetos descartados: {len(df_merged) - len(df_merged_clean)}")

    print("Cargando datos...")
    BP = torch.from_numpy(np.stack(df_merged_clean["BP"].values)).float()
    RP = torch.from_numpy(np.stack(df_merged_clean["RP"].values)).float()


    print("Centrando espectros...")
    BP = BP - BP.mean(dim=1, keepdim=True)
    RP = RP - RP.mean(dim=1, keepdim=True)

    batch_size = 256

    train_loader_bp = DataLoader(
        TensorDataset(BP, BP),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    train_loader_rp = DataLoader(
        TensorDataset(RP, RP),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    class SpectraConvAutoencoder(nn.Module):
        def __init__(self, input_len, latent_dim=latent_dim):
            super().__init__()

            self.encoder_cnn = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.1),

                nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.1),

                nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.1)
            )

            with torch.no_grad():
                dummy = torch.zeros(1, 1, input_len)
                encoded = self.encoder_cnn(dummy)
                self._enc_channels = encoded.shape[1]
                self._enc_len = encoded.shape[2]
                flattened = self._enc_channels * self._enc_len

            self.encoder_fc = nn.Linear(flattened, latent_dim)


            self.decoder_fc = nn.Linear(latent_dim, flattened)

            self.decoder_cnn = nn.Sequential(
                nn.ConvTranspose1d(
                    64, 32, kernel_size=5, stride=2, padding=2, output_padding=1
                ),
                nn.LeakyReLU(0.1),

                nn.ConvTranspose1d(
                    32, 16, kernel_size=5, stride=2, padding=2, output_padding=1
                ),
                nn.LeakyReLU(0.1),

                nn.ConvTranspose1d(
                    16, 1, kernel_size=5, stride=2, padding=2, output_padding=1
                )
            )

        def encode(self, x):
            h = self.encoder_cnn(x)
            h = h.view(h.size(0), -1)
            z = self.encoder_fc(h)
            return z

        def decode(self, z):
            h = self.decoder_fc(z)
            h = h.view(z.size(0), self._enc_channels, self._enc_len)
            x_rec = self.decoder_cnn(h)
            return x_rec

        def forward(self, x):
            z = self.encode(x)
            x_rec = self.decode(z)
            x_rec = x_rec[..., :x.shape[-1]]
            return x_rec, z

    def train_autoencoder(model, loader, n_epochs=100, lr=1e-3):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(n_epochs):
            total_loss = 0.0

            for x, _ in loader:
                x = x.to(device, non_blocking=True)
                x = x.unsqueeze(1)  # (batch, 1, 100)

                optimizer.zero_grad()
                x_rec, _ = model(x)
                loss = criterion(x_rec, x)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{n_epochs} "
                    f"Loss: {total_loss / len(loader):.3e}"
                )

    print("Creando modelos...")
    ae_bp = SpectraConvAutoencoder(bp_target, latent_dim).to(device)
    ae_rp = SpectraConvAutoencoder(rp_target, latent_dim).to(device)

    print("Entrenando autoencoder BP...")
    train_autoencoder(ae_bp, train_loader_bp, n_epochs=100)

    print("Entrenando autoencoder RP...")
    train_autoencoder(ae_rp, train_loader_rp, n_epochs=100)

    print("Calculando representaciones latentes y errores...")
    ae_bp.eval()
    ae_rp.eval()

    print("Guardando modelos...")
    torch.save(ae_bp.state_dict(), f"{OUTPUT_PATH}/ae_bp_conv_{latent_dim}d_latent.pth")
    torch.save(ae_rp.state_dict(), f"{OUTPUT_PATH}/ae_rp_conv_{latent_dim}d_latent.pth")

    with torch.no_grad():
        BP_gpu = BP.to(device)
        x_rec_bp, BP_latent = ae_bp(BP_gpu.unsqueeze(1))
        x_rec_bp = x_rec_bp.squeeze(1)
        RP_gpu = RP.to(device)
        x_rec_rp, RP_latent = ae_rp(RP_gpu.unsqueeze(1))
        x_rec_rp = x_rec_rp.squeeze(1)

    BP_latent = BP_latent.cpu()
    RP_latent = RP_latent.cpu()

    print("Guardando representaciones latentes...")

    df_autoenc = pd.DataFrame()
    for i in range(latent_dim):
        df_autoenc[f"BP_latent_{i+1}"] = BP_latent[:, i].numpy()
        df_autoenc[f"RP_latent_{i+1}"] = RP_latent[:, i].numpy()
    df_autoenc["source_id"] = df_merged_clean["source_id"].values
    df_autoenc["MG"] = df_merged_clean["MG"].values
    df_autoenc["BP_RP"] = df_merged_clean["BP_RP"].values

    df_autoenc.to_parquet(f"{OUTPUT_PATH}/ae_conv_{latent_dim}d_latent.parquet", index=False)
    print(f"Archivo 'ae_conv_{latent_dim}d_latent.parquet' generado con éxito.")
    save_pairplots(df_autoenc, "AE_CONV", latent_dim)

def process_gaia_sources():
    df_gaia = pd.read_csv(f"{GAIA_SOURCES_FILENAME}")
    df_gaia.to_parquet(f"{OUTPUT_PATH}/{GAIA_SOURCES_PARQUET_FILENAME}", index=False)
    return df_gaia

def process_spectra_files():
    files = []
    pattern = re.compile(SPECTRA_FILES_REGEX)
    for filename in os.listdir(SPECTRA_PATH):
        if pattern.match(filename):
            files.append(f"{SPECTRA_PATH}/{filename}")

    files.sort()
    print(f"Encontrados {len(files)} archivos para procesar")

    processed_dataframes = []
    successful_files = 0
    total_pairs = 0

    for file in tqdm(files, desc="Procesando archivos CSV", unit="archivo"):
        df = pd.read_csv(file)

        processed_rows = []
        for i in range(0, len(df), 2):
            row_bp = df.iloc[i]
            row_rp = df.iloc[i + 1]
            assert row_bp['source_id'] == row_rp['source_id']

            processed_row = {
                'source_id': row_bp['source_id'],
                'BP': row_bp['flux'] if row_bp['xp'] == 'BP' else row_rp['flux'],
                'RP': row_bp['flux'] if row_bp['xp'] == 'RP' else row_rp['flux']
            }
            processed_rows.append(processed_row)

        processed_df = pd.DataFrame(processed_rows)
        processed_dataframes.append(processed_df)
        successful_files += 1
        total_pairs += len(processed_rows)


    print("Concatenando espectros...")
    df_spectra	 = pd.concat(processed_dataframes, ignore_index=True)
    df_spectra.drop_duplicates(subset=['source_id'], inplace=True)

    print("Convirtiendo datos literales...")
    df_spectra["BP"] = df_spectra["BP"].apply(ast.literal_eval)
    df_spectra["RP"] = df_spectra["RP"].apply(ast.literal_eval)

    print("Normalizando espectros...")
    df_spectra['BP'] = df_spectra['BP'].apply(lambda x: np.array(x) / np.sum(x))
    df_spectra['RP'] = df_spectra['RP'].apply(lambda x: np.array(x) / np.sum(x))

    print("Guardando espectros en disco...")
    output_file = f"{OUTPUT_PATH}/{SPECTRA_PARQUET_FILENAME}"
    df_spectra.to_parquet(output_file, index=False)
    return df_spectra


def reduce_spectra_by_wavelength(df_spectra):
    wl_bp = pwl_to_wl('BP', SPECTRA_SAMPLING)
    wl_min_bp, wl_max_bp = (330, 800)
    valid_indices_bp = np.where((wl_bp >= wl_min_bp) & (wl_bp <= wl_max_bp))[0]
    df_spectra['BP'] = df_spectra['BP'].apply(lambda x: np.array(x)[valid_indices_bp].tolist())
    df_spectra['RP'] = df_spectra['RP'].apply(lambda x: np.array(x)[1::2].tolist())
    
    return df_spectra


####################################
#              Main                #
####################################

SELECTED_ALGORITHM, LATENT_DIM = parse_cli_args()
print(f"Algoritmo seleccionado: {SELECTED_ALGORITHM}")
print(f"Dimensión latent_dim: {LATENT_DIM}")

#Check if exist spectra parquet, if not, process the spectra files
df_spectra = pd.DataFrame()
if not os.path.exists(f"{OUTPUT_PATH}/{SPECTRA_PARQUET_FILENAME}"):
    print("No se encontró el archivo de espectros en disco. Procesando archivos CSV...")
    df_spectra = process_spectra_files()
else:
    print("Se encontró el archivo Parquet de espectros en disco. Cargando archivo de espectros...")
    df_spectra = pd.read_parquet(f"{OUTPUT_PATH}/{SPECTRA_PARQUET_FILENAME}")

df_sources = pd.DataFrame()
if not os.path.exists(f"{OUTPUT_PATH}/{GAIA_SOURCES_PARQUET_FILENAME}"):
    print("No se encontró el archivo descriptivo de fuentes de Gaia en disco. Procesando archivo CSV...")
    df_sources = process_gaia_sources()
else:
    print("Se encontró el archivo Parquet de fuentes de Gaia en disco. Cargando archivo de fuentes de Gaia...")
    df_sources = pd.read_parquet(f"{OUTPUT_PATH}/{GAIA_SOURCES_PARQUET_FILENAME}")

df_merged = pd.DataFrame()
if not os.path.exists(f"{OUTPUT_PATH}/{MERGED_DATA_PARQUET_FILENAME}"):
    print("No se encontró el archivo de datos combinados en disco. Filtrando y combinando datos...")
    print("Calculando columnas magnitud y color...")
    df_sources["BP_RP"] = df_sources["phot_bp_mean_mag"] - df_sources["phot_rp_mean_mag"]
    df_sources["MG"] = df_sources["phot_g_mean_mag"] + 5 - 5 * np.log10(df_sources["r_med_geo"])

    print("Reduciendo espectros por longitud de onda")
    df_spectra = reduce_spectra_by_wavelength(df_spectra)

    print("Combinando dataframes...")
    df_merged = pd.merge(df_sources, df_spectra, on="source_id", how="inner")

    print("Filtrando fuentes con BP-RP < 3.3...")
    df_merged = df_merged[df_merged['BP_RP'] < 3.3]
    filtered_sources = len(df_spectra) - len(df_merged)

    print(f"Actualizando ficheros parquet de espectros y fuentes de Gaia tras filtrado...")
    df_spectra = df_spectra[df_spectra['source_id'].isin(df_merged['source_id'])]
    df_sources = df_sources[df_sources['source_id'].isin(df_merged['source_id'])]
    df_spectra.to_parquet(f"{OUTPUT_PATH}/{SPECTRA_PARQUET_FILENAME}")
    df_sources.to_parquet(f"{OUTPUT_PATH}/{GAIA_SOURCES_PARQUET_FILENAME}")

    print("Guardando en fichero parquet...")
    df_merged.to_parquet(f"{OUTPUT_PATH}/{MERGED_DATA_PARQUET_FILENAME}")

else:
    print("Se encontró el archivo de datos combinados en disco. Cargando archivo de datos combinados...")
    df_merged = pd.read_parquet(f"{OUTPUT_PATH}/{MERGED_DATA_PARQUET_FILENAME}")

#######################################
#              Plotting               #
#######################################
plt.rcParams['text.usetex'] = False

def median_spectrum(spectra):
    # Convert all spectrum entries to numpy arrays of float type
    np_spectra = []
    for s_val in spectra.values:
        # Ensure s_val is iterable and not empty before converting to array
        if isinstance(s_val, (list, np.ndarray)) and len(s_val) > 0:
            np_spectra.append(np.array(s_val, dtype=float))
        # Handle cases where s_val might be empty list or None, by appending nan array or skipping
        elif isinstance(s_val, (list, np.ndarray)) and len(s_val) == 0:
            # If an empty list, consider it as a single NaN, to ensure a length
            np_spectra.append(np.array([np.nan], dtype=float))
        # For any other unexpected type, it will be skipped from np_spectra

    if not np_spectra:
        # If no valid spectra were found in the group, return an empty array or NaNs
        return np.array([np.nan])

    max_len = max(s.shape[0] for s in np_spectra)

    padded_spectra = []
    for s_arr in np_spectra:
        if s_arr.shape[0] < max_len:
            # Pad with NaN to match max_len, ensuring float type
            padded_s = np.pad(s_arr, (0, max_len - s_arr.shape[0]), 'constant', constant_values=np.nan)
            padded_spectra.append(padded_s)
        else:
            padded_spectra.append(s_arr)

    # All arrays in padded_spectra should now have max_len
    return np.nanmedian(np.stack(padded_spectra), axis=0)

print("Cargando eje de longitudes de onda filtradas...")
wl_bp = pwl_to_wl('BP',SPECTRA_SAMPLING)
wl_min_bp, wl_max_bp = (330, 800)
valid_indices_bp = np.where((wl_bp >= wl_min_bp) & (wl_bp <= wl_max_bp))[0]
wl_bp = wl_bp[valid_indices_bp]
bp_sort_idx = np.argsort(wl_bp)
wl_bp = wl_bp[bp_sort_idx]

wl_rp = pwl_to_wl('RP',SPECTRA_SAMPLING)
wl_rp = wl_rp[1::2]

bins = 4
color_bins = np.linspace(1.7, 3.3, bins + 1)
mg_bins = np.linspace(9.5, 11.0, bins + 1)

df_merged['color_bin'] = pd.cut(df_merged['BP_RP'], bins=color_bins)
df_merged['mg_bin'] = pd.cut(df_merged['MG'], bins=mg_bins)

bp_medians = (
    df_merged
    .groupby(['mg_bin', 'color_bin'], observed=True)['BP']
    .apply(median_spectrum)
)
rp_medians = (
    df_merged
    .groupby(['mg_bin', 'color_bin'], observed=True)['RP']
    .apply(median_spectrum)
)

# Diagnóstico de fuentes por celda (mg_bin, color_bin)
# print("Conteos por bin:")
# print(df_merged.groupby(['mg_bin', 'color_bin'], observed=True).size().unstack(fill_value=0))

cmap = cm.rainbow
norm = colors.Normalize(vmin=color_bins.min(), vmax=color_bins.max())

fig, axes = plt.subplots(bins, 1, figsize=(6, 2.8 * bins), sharex=True)

for i, mag_bin in enumerate(bp_medians.index.levels[0]):
    ax = axes[i]

    for color_bin in bp_medians.index.levels[1]:
        spec_bp = bp_medians.loc[(mag_bin, color_bin)]
        spec_bp_plot = spec_bp[bp_sort_idx]
        color_val = color_bin.mid
        color = cmap(norm(color_val))

        ax.plot(wl_bp, spec_bp_plot, color=color, lw=1.5)

    ax.set_ylabel('Flujo')
    ax.set_title(f'MG ∈ {mag_bin}', fontsize=10)
    ax.grid(True)

    if i == bins - 1:
        ax.set_xlabel('Longitud de onda [nm]')
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.subplots_adjust(left=0.1, right=0.85, hspace=0.25)
cax = fig.add_axes([0.87, 0.15, 0.025, 0.7])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('BP − RP')
plt.suptitle('Espectros BP (medianas) por rangos de magnitud y color')
plt.savefig(f"{OUTPUT_PATH}/bp_medians.png", dpi=200, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(bins, 1, figsize=(6, 2.8 * bins), sharex=True)

for i, mag_bin in enumerate(rp_medians.index.levels[0]):
    ax = axes[i]

    for color_bin in rp_medians.index.levels[1]:
        spec_rp = rp_medians.loc[(mag_bin, color_bin)]
        color_val = color_bin.mid
        color = cmap(norm(color_val))

        ax.plot(wl_rp, spec_rp, color=color, lw=1.5)

    ax.set_ylabel('Flujo')
    ax.set_title(f'MG ∈ {mag_bin}', fontsize=10)
    ax.grid(True)

    if i == bins - 1:
        ax.set_xlabel('Longitud de onda [nm]')
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.subplots_adjust(left=0.1, right=0.85, hspace=0.25)
cax = fig.add_axes([0.87, 0.15, 0.025, 0.7])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('BP − RP')
plt.suptitle('Espectros RP (medianas) por rangos de magnitud y color')
plt.savefig(f"{OUTPUT_PATH}/rp_medians.png", dpi=200, bbox_inches="tight")
plt.close()

if SELECTED_ALGORITHM == "UMAP":
    run_umap(LATENT_DIM, df_merged, wl_bp, wl_rp)
elif SELECTED_ALGORITHM == "PCA":
    run_pca(LATENT_DIM)
elif SELECTED_ALGORITHM == "AE":
    run_ae(LATENT_DIM)
elif SELECTED_ALGORITHM == "AE_CONV":
    run_ae_conv(LATENT_DIM, df_merged, wl_bp, wl_rp)
else:
    raise ValueError(f"Algoritmo no soportado: {SELECTED_ALGORITHM}")