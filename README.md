# README - script_master_work.py

Este documento explica como preparar el entorno para ejecutar `script_master_work.py`, validar los ficheros de entrada y lanzar el procesamiento.

## 1) Ficheros implicados

- Script principal: `script_master_work.py`
- Script de preparacion: `setup_script_master_work.sh`

## 2) Que hace el script de preparacion

El script `setup_script_master_work.sh` automatiza tres tareas:

1. Instala dependencias Python necesarias para `script_master_work.py`.
2. Crea el directorio de salida esperado por el programa: `output_test`.
3. Comprueba que los CSV de entrada estan en su sitio:
   - `sources_gaia_task1.csv`
   - Carpeta `spectra_chunks_task1` con archivos `output_spectra_###.csv`

Si falta algun archivo/directorio, el script termina con error y mensaje descriptivo.

## 3) Requisitos previos

- Entorno bash (por ejemplo, WSL o Git Bash).
- Python 3 con `pip`.
- Permisos para instalar paquetes Python en el entorno activo.

## 4) Estructura esperada

En `C:\Users\yeyos` (o `/mnt/c/Users/yeyos` desde bash) deben existir:

- `script_master_work.py`
- `sources_gaia_task1.csv`
- `spectra_chunks_task1/`
  - `output_spectra_000.csv`
  - `output_spectra_001.csv`
  - `...`

## 5) Como ejecutar la preparacion

Desde bash:

```bash
bash /mnt/c/Users/yeyos/setup_script_master_work.sh
```

## 6) Como ejecutar el script principal

El script principal requiere dos argumentos:

```bash
python3 script_master_work.py <ALGORITMO> <BOTTLENECK>
```

Valores permitidos:

- `ALGORITMO`: `UMAP`, `PCA`, `AE`, `AE_CONV`
- `BOTTLENECK`: `3`, `5`, `10`

Ejemplo:

```bash
cd /mnt/c/Users/yeyos
python3 script_master_work.py UMAP 3
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

- Si vuelves a ejecutar el script, reutiliza los `.parquet` si ya existen.
- El script de setup incluye `pyarrow` porque el programa usa `to_parquet`/`read_parquet`.
