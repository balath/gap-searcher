# SPEctra LAtent REpresentations BUilder Script (SPELAREBUS)

Preparación del entorno para ejecutar `spelarebus.py`, validar los ficheros de entrada y lanzar el procesamiento.

## 1) Ficheros implicados

- Script principal: `spelarebus.py`
- Script de preparacion: `setup_spelarebus.sh`

## 2) Que hace el script de preparacion

El script `setup_spelarebus.sh` automatiza tres tareas:

1. Instala dependencias Python necesarias para `spelarebus.py`.
2. Crea el directorio de salida esperado por el programa: `output_files`.
3. Comprueba que los CSV de entrada estan en su sitio:
   - `sources_gaia.csv`
   - Carpeta `spectra_chunks` con archivos `output_spectra_###.csv`

Si falta algun archivo/directorio, el script termina con error y mensaje descriptivo.

## 3) Requisitos previos

- Entorno bash
- Python 3 con `pip`.
- Permisos para instalar paquetes Python en el entorno activo.

## 4) Estructura esperada

En el directorio raíz donde se ejecute spelarebus deben existir:

- `spelarebus.py`
- `sources_gaia.csv`
- `spectra_chunks/`
  - `output_spectra_000.csv`
  - `output_spectra_001.csv`
  - `...`

## 5) Como ejecutar la preparacion

Desde bash:

```bash
./setup_spelarebus.sh
```

## 6) Como ejecutar el script principal

El script principal requiere dos argumentos:

```bash
python3 spelarebus.py <ALGORITMO> <LATENT_DIM>
```

Valores permitidos:

- `ALGORITMO`: `UMAP`, `PCA`, `AE`, `AE_CONV`
- `LATENT_DIM`: `3`, `5`, `10`

Ejemplo:

```bash
python3 spelarebus.py UMAP 3
```

## 7) Salidas generadas

El programa escribe sus resultados en `output_test/`, incluyendo:

- Parquet intermedios:
  - `spectra.parquet`
  - `sources.parquet`
  - `merged_data.parquet`
- Graficos:
  - `bp_medians.png`
  - `rp_medians.png`
  - pairplots segun algoritmo
- Resultados de reduccion segun algoritmo (por ejemplo, `umap_3d_latent.parquet`).

## 8) Notas

- Si vuelves a ejecutar el script, reutiliza los `.parquet` previamente creados.
