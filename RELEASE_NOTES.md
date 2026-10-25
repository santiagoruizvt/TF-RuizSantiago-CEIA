# Release Notes

## v0.1.1 — Front-end facial: detección, ROIs y recortes

Primera etapa que procesa los videos. La comparten el pipeline clásico, a través de las trazas RGB, y el de DL, a través de los recortes de cara.

### Agregado
- `rppg.face.landmarker`: MediaPipe Face Landmarker (478 landmarks) en modo VIDEO.
  - Fuerza timestamps estrictamente crecientes, porque UBFC trae timestamps repetidos.
  - Suaviza los landmarks entre frames y reinicia el suavizado cuando un frame no tiene cara.
  - Descarga el modelo verificando su SHA-256.
- `rppg.face.roi`: ROIs de frente y mejillas definidas por landmarks, máscara de piel opcional en YCrCb y recorte cuadrado de la cara para el DL.
- `rppg.face.extract` y `scripts/extract_traces.py`:
  - una sola pasada por video;
  - guarda las trazas RGB por ROI y los recortes de cara en `data/processed/`;
  - usa cache invalidada por hash de la configuración;
  - escribe un resumen de tasa de detección y tiempo de procesamiento.
- `configs/face.yaml`: modelo, índices de las ROIs, suavizado, máscara de piel y recortes.
- `rppg.utils.config`: lectura de `.yaml` y hash de configuración.
- Tests: ROIs con colores exactos, recortes en bordes, suavizado y reloj.
  - La extracción completa sobre el dataset sintético recupera la HR conocida.
  - El detector se prueba sobre una imagen de dominio público (`skimage.data.astronaut`) y, si está disponible, sobre UBFC real.
- Notebook `02_face_frontend`: recorre detector, ROIs, máscara de piel, recortes y extracción con figuras, y cierra con una primera estimación de HR con el canal verde.
- CI: guarda el modelo de MediaPipe en la caché de GitHub.

### Resultados sobre UBFC-rPPG
- Los 42 sujetos procesados.
- Detección facial en el **100 % de los frames** de todos los sujetos (requerimiento: ≥ 95 %).
- Mediana de **48 s de procesamiento por minuto de video** en CPU, con un máximo de 50 s (requerimiento: ≤ 2 min).
- Trazas y recortes de 128×128 ocupan **3,2 GB** en total (~78 MB por sujeto), lo que permite entrenar en Colab sin subir los ~65 GB de video crudo.
- El suavizado de landmarks con `alpha = 0.5` no tiene un efecto medible. Los landmarks ya son estables (~0,4 px por frame) y el ruido dominante en las trazas es el movimiento y la iluminación.
- Vistazo preliminar con el canal verde de la frente, que no es el baseline oficial: MAE 11,1 bpm, RMSE 22,7 bpm y Pearson 0,53 en todas las ventanas. 24 de los 42 sujetos quedan con MAE ≤ 5 bpm. CHROM y POS llegan en la v0.2.0.

### Decisiones
- RR: COHFACE queda postergado. La RR se estima sobre UBFC-rPPG como resultado exploratorio, sin métricas formales.

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

