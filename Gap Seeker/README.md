# Visualización astrofísica – dimensiones latentes y diagrama color-magnitud

Aplicación Bokeh para explorar datos astrofísicos en parquet: pairplot de dimensiones latentes con selección por lazo y diagrama color-magnitud absoluta (BP_RP vs MG) actualizable.

## Requisitos del parquet

- Columnas obligatorias: `source_id`, `BP_RP`, `MG`
- Dimensiones para el pairplot: columnas que empiecen por `BP_` o `RP_` (p. ej. BP_G, RP_G, BP_RP, etc.)

## Instalación

```bash
cd astro_app
pip install -r requirements.txt
```

## Uso

1. Arranca el servidor y abre la app en el navegador (por defecto `http://localhost:5006/app`).

2. **Cargar parquet**: escribe la ruta al archivo **relativa al directorio donde está la app** (p. ej. `datos.parquet` o `subcarpeta/datos.parquet`) y pulsa **Cargar desde ruta**. El archivo debe estar en esa carpeta o en una subcarpeta.

3. Elige dos dimensiones en los desplegables y pulsa **Show pairplot**.

4. Selecciona puntos en el pairplot con la herramienta **Lasso** (ícono de lazo en la barra de herramientas).

5. Pulsa **Actualizar diagrama color-magnitud** para rellenar el diagrama BP_RP vs MG solo con las fuentes seleccionadas.

## Tamaños

- Pairplot: 500×500 px; puntos size 0.5, alpha 1.
- Diagrama color-magnitud: 900×500 px; puntos size 0.5, alpha 1.

Si con muchos puntos va lento, reduce el número de filas del parquet (por ejemplo un subconjunto o un muestreo) antes de cargarlo.
