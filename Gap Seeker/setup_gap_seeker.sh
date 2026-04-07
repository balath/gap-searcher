#!/usr/bin/env bash
set -euo pipefail

# Configuracion principal
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$APP_DIR/gap_seeker.py"
REQ_PATH="$APP_DIR/requirements.txt"
MEDIANS_CALCULATOR_PATH="$APP_DIR/precompute_spectra_medians.py"

REQUIRED_SPECTRA_PARQUET="$APP_DIR/reduced_spectra.parquet"
REQUIRED_WL_BP="$APP_DIR/filtered_wl_bp.npy"
REQUIRED_WL_RP="$APP_DIR/filtered_wl_rp.npy"
REQUIRED_MEDIANS_PARQUET="$APP_DIR/spectra_medians.parquet"

echo "==> Directorio de la app: $APP_DIR"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "ERROR: No existe el script en $SCRIPT_PATH"
  exit 1
fi

if [[ ! -f "$REQ_PATH" ]]; then
  echo "ERROR: No existe requirements.txt en $REQ_PATH"
  exit 1
fi

if [[ ! -f "$MEDIANS_CALCULATOR_PATH" ]]; then
  echo "ERROR: No existe el calculador de medianas en $MEDIANS_CALCULATOR_PATH"
  exit 1
fi

# 1) Instalar dependencias Python
echo "==> Instalando dependencias..."
python3 -m pip install --upgrade pip
python3 -m pip install -r "$REQ_PATH"

# 2) Validar archivos base requeridos
echo "==> Validando archivos base requeridos..."
validation_errors=()

if [[ ! -f "$REQUIRED_SPECTRA_PARQUET" ]]; then
  validation_errors+=("Falta $REQUIRED_SPECTRA_PARQUET")
fi

if [[ ! -f "$REQUIRED_WL_BP" ]]; then
  validation_errors+=("Falta $REQUIRED_WL_BP")
fi

if [[ ! -f "$REQUIRED_WL_RP" ]]; then
  validation_errors+=("Falta $REQUIRED_WL_RP")
fi

if (( ${#validation_errors[@]} > 0 )); then
  echo
  echo "Se detectaron errores de validacion:"
  for err in "${validation_errors[@]}"; do
    echo "  - ERROR: $err"
  done
  exit 1
fi

# 3) Calcular medianas precalculadas
echo "==> Calculando medianas precalculadas..."
python3 "$MEDIANS_CALCULATOR_PATH"

# 4) Verificar salida de medianas
echo "==> Verificando salida de medianas..."
if [[ ! -f "$REQUIRED_MEDIANS_PARQUET" ]]; then
  echo "ERROR: No se genero $REQUIRED_MEDIANS_PARQUET"
  exit 1
fi

echo "OK: reduced_spectra.parquet encontrado."
echo "OK: filtered_wl_bp.npy y filtered_wl_rp.npy encontrados."
echo "OK: spectra_medians.parquet encontrado."
echo "Todo listo."
echo
echo "Ejemplo de ejecucion:"
echo "  cd \"$APP_DIR\""
echo "  bokeh serve --show gap_seeker.py"
