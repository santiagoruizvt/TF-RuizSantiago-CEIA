# Release Notes

## v0.1.0 — Núcleo DSP, loader de UBFC-rPPG y CI

Base común a los enfoques clásico y de deep learning. Todavía no incluye detección facial ni métodos rPPG.

### Agregado
- `rppg.dsp`, compartido entre ambos enfoques:
  - pasa-banda Butterworth de fase cero;
  - remoción de tendencia lineal y por *smoothness priors*;
  - remuestreo a grilla uniforme con timestamps y marcado de huecos;
  - estimación espectral en ventanas deslizantes (periodograma, interpolación parabólica del pico, SNR).
- `rppg.eval.metrics`: MAE, RMSE y Pearson, descartando ventanas inválidas.
- `rppg.data.ubfc`: loader de UBFC-rPPG (DATASET_2), con lectura de video frame a frame.
- Tests unitarios con señales sintéticas y un dataset sintético con formato UBFC-rPPG. Validan la cadena completa de frames a HR sin el dataset real.
- CI con lint, formato (ruff) y pytest en cada PR.
- Entorno con uv y Python 3.12; PyTorch para CPU como extra opcional `dl`.
- Notebook `01_ubfc_exploration`: formato y calidad del ground truth.

### Decisiones
- Detector facial: MediaPipe Face Landmarker.
- RR por tres vías intercambiables: RIIV, RIAV y RIFV.
- Ventanas: HR de 10 s con paso de 1 s; RR de 30 s con paso de 5 s; además, una estimación por video.
- RR: exploratoria en UBFC-rPPG y métrica formal contra cinturón respiratorio en COHFACE, sujeta a obtener la licencia.
- La HR de referencia se calcula desde la PPG del oxímetro con el mismo estimador que se aplica al rPPG. La HR que reporta el equipo tiene cortes y no se usa.

## Workflows compartidos

- Se incorporan los workflows de pre-release y release automático gestionados por santiagoruizvt/sr-workflows.

