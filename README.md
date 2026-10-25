# TF-RuizSantiago-CEIA
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
- `data/`: datasets y caches (no versionados).
  - UBFC-rPPG va en `data/raw/subjectN/`.
  - Las trazas y los recortes extraídos quedan en `data/processed/`.
- `models/`: modelos descargados (no versionados). El de MediaPipe se baja solo y se verifica su hash.

## Uso

```bash
# Trazas RGB por ROI y recortes de cara de todos los videos (~45 s por minuto de video, en CPU)
uv run python scripts/extract_traces.py
```

## Versionado

- **Ramas:** `main` siempre funciona: los tests pasan. Cada tarea se hace en una rama corta (`feat/...`, `fix/...`, `docs/...`, `exp/...`) y se integra con un Pull Request con *squash merge*.
- **Commits:** en español, con el prefijo de tipo de [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/) (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
- **Release notes:** cada PR a `main` **debe** agregar una sección nueva al principio de `RELEASE_NOTES.md`, en español. Las líneas agregadas pasan a ser el cuerpo del release en GitHub.
- **Versiones automáticas:** las gestionan los workflows compartidos de [`santiagoruizvt/sr-workflows`](https://github.com/santiagoruizvt/sr-workflows). Sus archivos en `.github/workflows/` no se editan a mano.
  - **Pre-release:** en cada PR verifica que el título tenga exactamente una etiqueta `#major`, `#minor` o `#patch`, y que el PR agregue líneas a `RELEASE_NOTES.md`.
  - **Release:** al mergear el PR, crea el tag `vX.Y.Z` y el GitHub Release, sumándole a la última versión el incremento indicado en el título.
  - **No crear tags a mano:** desordenarían el cálculo de la próxima versión.
- **Hitos:** el PR que completa un hito lleva `#minor`. Los PRs intermedios y las correcciones llevan `#patch` (por ejemplo `v0.1.1`, `v0.1.2` hasta llegar a `v0.2.0`).

  | Versión | Hito |
  |---|---|
  | `v0.1.0` | Base compartida (`dsp`, métricas, loader de UBFC) |
  | `v0.2.0` | Baseline clásico de HR (GREEN, CHROM, POS) con métricas sobre UBFC |
  | `v0.3.0` | RR clásica (exploratoria en UBFC; formal si llega COHFACE) |
  | `v0.4.0` | Modelo de DL entrenado y validado |
  | `v0.5.0` | Evaluación comparativa completa |
  | `v1.0.0` | Versión de la memoria final y la defensa |

  Las correcciones entre hitos incrementan el *patch* (`v0.2.1`).
- **Trazabilidad de resultados:** todo resultado que vaya a la memoria registra el commit o tag con que se generó.
