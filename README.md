# AdSVmTdVpC-CEIA
Este repositorio contiene todo lo relacionado con el trabajo final titulado "Análisis de Signos Vitales mediante Técnicas de Visión por Computadora", de la Carrera de Especialización en Inteligencia Artificial.

## Entorno

Requiere [uv](https://docs.astral.sh/uv/). Python 3.12 (fijado en `.python-version`).

```bash
uv sync               # pipeline clásico + herramientas de desarrollo (pytest, jupyter, ruff)
uv sync --extra dl    # además PyTorch (CPU) para la Etapa 2
uv run pytest         # tests
uv run jupyter lab    # notebooks
```

Si `uv` falla con `invalid peer certificate: UnknownIssuer` (antivirus o proxy que inspecciona TLS), agregar `--system-certs` a los comandos de uv.

## Estructura

- `src/rppg/`: código del proyecto como paquete instalable (`data`, `face`, `classical`, `dsp`, `eval`, `dl`, `utils`).
- `scripts/`: puntos de entrada reproducibles (extracción de trazas, ejecución, evaluación).
- `notebooks/`: exploración y figuras; importan desde `rppg`, no definen lógica propia.
- `configs/`: parámetros en `.yaml`.
- `data/`: datasets y caches (no versionados). UBFC-rPPG va en `data/raw/UBFC-rPPG/`.
- `docs/`: decisiones de diseño y borradores de la memoria.
