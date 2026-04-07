#!/usr/bin/env bash
set -euo pipefail

# Configuracion principal
SCRIPT_PATH="spelarebus.py"
BASE_DIR="$(dirname "$SCRIPT_PATH")"
OUTPUT_DIR="$BASE_DIR/output_files"
SPECTRA_DIR="$BASE_DIR/spectra_chunks"
SOURCES_CSV="$BASE_DIR/sources_gaia.csv"

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

validation_errors=()

if [[ ! -f "$SOURCES_CSV" ]]; then
  validation_errors+=("Falta $SOURCES_CSV")
fi

if [[ ! -d "$SPECTRA_DIR" ]]; then
  validation_errors+=("Falta el directorio $SPECTRA_DIR")
fi

shopt -s nullglob
spectra_files=("$SPECTRA_DIR"/output_spectra_[0-9][0-9][0-9].csv)
shopt -u nullglob

if (( ${#spectra_files[@]} == 0 )); then
  validation_errors+=("No hay CSV validos en $SPECTRA_DIR con patron output_spectra_###.csv")
fi

if (( ${#validation_errors[@]} > 0 )); then
  echo
  echo "Se detectaron errores de validacion:"
  for err in "${validation_errors[@]}"; do
    echo "  - ERROR: $err"
  done
  exit 1
fi

echo "OK: Encontrados ${#spectra_files[@]} archivos de espectros."
echo "OK: sources_gaia.csv encontrado."
echo "Todo listo."
echo
echo "Ejemplo de ejecucion:"
echo "  cd \"$BASE_DIR\""
echo "  python3 spelarebus.py UMAP 3"
