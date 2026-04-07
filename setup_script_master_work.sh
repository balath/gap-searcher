#!/usr/bin/env bash
set -euo pipefail

# Configuracion principal
SCRIPT_PATH="/mnt/c/Users/yeyos/script_master_work.py"
BASE_DIR="$(dirname "$SCRIPT_PATH")"
OUTPUT_DIR="$BASE_DIR/output_test"
SPECTRA_DIR="$BASE_DIR/spectra_chunks_task1"
SOURCES_CSV="$BASE_DIR/sources_gaia_task1.csv"

echo "==> Directorio base: $BASE_DIR"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "ERROR: No existe el script en $SCRIPT_PATH"
  exit 1
fi

# 1) Instalar dependencias Python
echo "==> Instalando dependencias..."
python3 -m pip install --upgrade pip
python3 -m pip install \
  pandas numpy tqdm gaiaxpy matplotlib seaborn pyarrow umap-learn torch

# 2) Crear directorio de salida esperado por el programa
echo "==> Creando directorio de salida..."
mkdir -p "$OUTPUT_DIR"
echo "Directorio listo: $OUTPUT_DIR"

# 3) Comprobar que los CSV estan en su sitio
echo "==> Validando archivos CSV de entrada..."

if [[ ! -f "$SOURCES_CSV" ]]; then
  echo "ERROR: Falta $SOURCES_CSV"
  exit 1
fi

if [[ ! -d "$SPECTRA_DIR" ]]; then
  echo "ERROR: Falta el directorio $SPECTRA_DIR"
  exit 1
fi

shopt -s nullglob
spectra_files=("$SPECTRA_DIR"/output_spectra_[0-9][0-9][0-9].csv)
shopt -u nullglob

if (( ${#spectra_files[@]} == 0 )); then
  echo "ERROR: No hay CSV validos en $SPECTRA_DIR con patron output_spectra_###.csv"
  exit 1
fi

echo "OK: Encontrados ${#spectra_files[@]} archivos de espectros."
echo "OK: sources_gaia_task1.csv encontrado."
echo "Todo listo."
echo
echo "Ejemplo de ejecucion:"
echo "  cd \"$BASE_DIR\""
echo "  python3 script_master_work.py UMAP 3"
