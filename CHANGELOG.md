# Changelog

Todos los cambios relevantes del proyecto se registran en este archivo.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el proyecto usa [Versionado Semántico](https://semver.org/lang/es/). Mientras la versión sea 0.y.z, cada versión menor corresponde a un hito técnico del plan de trabajo (ver "Versionado" en el README).

## [No publicado]

### Agregado
- Paquete `rppg` instalable con uv, en Python 3.12. PyTorch para CPU queda como extra opcional `dl`.
- `rppg.dsp`, compartido entre el pipeline clásico y el de DL:
  - pasa-banda Butterworth de fase cero;
  - remoción de tendencia lineal y por *smoothness priors*;
  - remuestreo a grilla uniforme con detección de huecos;
  - estimación espectral en ventanas deslizantes, con interpolación parabólica del pico y SNR.
- `rppg.eval.metrics`: MAE, RMSE y Pearson. Descartan los pares inválidos.
- `rppg.data.ubfc`: loader de UBFC-rPPG (DATASET_2). Lee la PPG, la HR del oxímetro y los timestamps, y devuelve los frames de a uno.
- Tests unitarios con señales sintéticas.
- Dataset sintético con formato UBFC-rPPG (`tests/conftest.py`):
  - HR conocida, timestamps irregulares con un hueco y HR de oxímetro con cortes;
  - valida el loader y la cadena frames → HR sin el dataset real.
  - Los mismos chequeos de formato corren también sobre la copia real si está disponible.
- CI con GitHub Actions: lint (ruff), formato y tests en cada PR y en cada push a `main`.
- Comentarios, docstrings y notebooks en español.
- Notebook `01_ubfc_exploration`: formato del ground truth, irregularidad de los timestamps, cortes en la HR del oxímetro y comparación con la HR estimada desde la PPG.

### Decisiones
- D1: detector facial MediaPipe Face Landmarker.
- D2: la RR se obtiene por las vías RIIV, RIAV y RIFV, intercambiables.
- D3: ventanas de evaluación de 10 s con paso de 1 s para HR, y de 30 s con paso de 5 s para RR, además de una estimación global por video.
- D4: RR exploratoria sobre UBFC-rPPG y métrica formal contra un cinturón respiratorio en COHFACE (alternativa: MAHNOB-HCI), sujeta a obtener la licencia.
- La HR de referencia se calcula desde la PPG del oxímetro con el mismo estimador que se aplica al rPPG. No se usa la HR que reporta el equipo.
