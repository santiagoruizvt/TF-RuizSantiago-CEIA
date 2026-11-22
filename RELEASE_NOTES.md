# Release Notes

## v0.1.8 — PhysNet: subject37 a train, sobremuestreo por HR y recorte temporal

Prepara las corridas de Colab que implementan las sugerencias del director (CLAUDE.md §10.5 y §10.7). Todavía no hay resultados: se agregan cuando se ejecute la notebook 03.

### Cambiado
- **Partición:** subject37 pasa de validación a train. Quedan 29 sujetos de train, 4 de validación (subject8, 12, 27 y 44) y los mismos 9 de test.
  - Es una edición manual de `configs/splits.yaml`, no una regeneración.
  - Un test nuevo fija la lista de test, para que no cambie por accidente.
  - Validación se queda sin sujetos con HR baja; el control pasa a ser subject23, en test.
- `configs/physnet.yaml`: la corrida por defecto pasa a llamarse `physnet_v3`. Sus métricas de validación no son comparables con las de `physnet_v2`, porque el conjunto de validación cambió.

### Agregado
- **Sobremuestreo por HR** (`augment.speed_mode: hr_balanced`, apagado por defecto):
  - en lugar de una velocidad uniforme, se elige una HR objetivo uniforme en `target_hr_range` (45 a 150 bpm) y la velocidad es objetivo / HR del tramo, recortada a `speed_range`;
  - la HR de cada tramo se estima una vez sobre la PPG del oxímetro, en una ventana de 10 s centrada en el tramo y con el estimador del protocolo (`rppg.dl.dataset.chunk_hr`);
  - motivo: con la partición nueva, los 829 tramos de train tienen HR de 53 a 135 bpm, pero solo 55 están por debajo de 80. La velocidad uniforme (0,6 a 1,4) amplía el rango, pero no cambia dónde se concentran los ejemplos.
- **Recorte temporal aleatorio** (`augment.temporal_jitter`, apagado por defecto): el inicio de cada tramo se desplaza al azar dentro de `train_stride`, en lugar de caer siempre en la misma grilla. Si el tramo desplazado tiene frames sin cara o se sale del video, se usa el original.
- Las dos transformaciones cambian video y PPG juntos. Transformar solo la señal enseñaría una correspondencia falsa (ver §10.7).
- Notebook 03: la corrida de referencia `physnet_v3` y los overrides de `physnet_v3_hrbal` y `physnet_v3_jitter`.
- Tests: estimación de HR por tramo; distribución uniforme de la HR objetivo; recorte a `speed_range`; video y etiqueta llevados a la HR objetivo; recorte temporal dentro del paso y respetando la detección facial.

### Plan de corridas (Colab, desde el tag v0.1.8)
1. `physnet_v3`: configuración por defecto con la partición nueva (referencia).
2. `physnet_v3_hrbal`: `augment.speed_mode=hr_balanced augment.speed_p=0.8 augment.speed_range=[0.4,2.0]`.
3. `physnet_v3_jitter`: `augment.temporal_jitter=true`.

Cada una cambia una sola cosa respecto de la referencia. La que mejore se combina, y la configuración final se entrena con 3 semillas para cerrar la tarea 3.4.

## v0.1.7 — PhysNet: promedio en inferencia

Primera sugerencia del director (§10.7) que no necesita re-entrenar: armar la señal del video con tramos solapados y promediados.

### Agregado
- `predict_signal` acepta dos opciones nuevas:
  - `stride`: con un paso menor que el tramo, combina los tramos solapados por promedio ponderado (*overlap-add*), con pesos de Hann sin ceros que bajan el peso de los bordes;
  - `flip_average`: promedia además con la predicción sobre el video espejado.
- `configs/physnet.yaml`, sección `infer`: `stride: 64` (solapamiento del 50 %) y `flip_average: false`. La usan la validación de cada época y `scripts/evaluate_dl.py`.
  - Los checkpoints sin esa sección (hasta la v0.1.6) se siguen evaluando sin solapamiento.
- `scripts/evaluate_dl.py`: los flags `--stride` y `--flip-average` pisan la config del checkpoint. Si la cambian, los archivos de salida llevan un sufijo (p. ej. `val_s64_summary.csv`), y `<split>_eval_info.json` registra las opciones usadas.
- Tests de cobertura de tramos, pesos, espejado y compatibilidad con configs viejas.

### Resultados
Mismo modelo (`physnet_v2`, época 5, entrenado con `5a477a1`), evaluado en validación en CPU y sin re-entrenar. Solo cambia cómo se arma la señal. El test sigue sin usarse.

| Inferencia | MAE | RMSE | Pearson | subject37 |
|---|---|---|---|---|
| Sin solapamiento (hasta la v0.1.6) | 5,34 | 16,14 | 0,68 | 22,96 |
| Solo espejado | 4,57 | 14,75 | 0,73 | 19,22 |
| **Solapamiento 50 %** | **0,49** | **0,94** | **1,00** | **0,40** |
| Solapamiento 50 % + espejado | 0,48 | 0,93 | 1,00 | 0,41 |
| Solapamiento 75 % + espejado | 0,48 | 0,92 | 1,00 | 0,40 |

MAE y RMSE en bpm, 287 ventanas de 10 s con paso de 1 s. Los otros cuatro sujetos también mejoran un poco: MAE medio de 0,79 a 0,52 bpm.

### Interpretación
- **Lo que corrige el error son los pesos de Hann, no el promediado en sí.** En la ventana de 32 a 42 s de subject37, la relación entre la potencia en la fundamental (~55 bpm) y en el armónico (~110 bpm) es:

  | Señal | Fundamental / armónico |
  |---|---|
  | Sin solapamiento | 0,92 (gana el armónico) |
  | Dos particiones desfasadas, promedio uniforme | 0,94 (gana el armónico) |
  | Solapamiento con pesos de Hann | 1,24 (gana la fundamental) |
  | Referencia | 5,11 |

  Los bordes de cada tramo, donde la red tiene menos contexto temporal, aportan sobre todo contenido en el armónico.
- **El margen es fino.** La fundamental gana 1,24 a 1, no 5 a 1 como en la referencia. El modelo sigue sin representar bien las HR bajas; el promedio en inferencia inclina un empate. Por eso se mantienen el sobremuestreo por HR y el cambio de partición del paso siguiente.
- El espejado casi no aporta cuando hay solapamiento (0,01 bpm) y duplica el costo, así que queda apagado.
- Limitaciones:
  - Una sola semilla.
  - La configuración se eligió mirando validación.
  - El efecto en HR bajas depende de un único sujeto, y subject37 pasa a train en la próxima versión (§10.5). El control de HR baja va a ser subject23, en test.

### Cambiado
- La validación de cada época usa ahora la sección `infer`, así que las próximas corridas eligen la época con la señal solapada. Los historiales de `physnet_v2` no son comparables época a época con los nuevos; la corrida de referencia con la partición nueva se vuelve a hacer de todos modos.
- `CLAUDE.md` §8: fila de R4 con esta evidencia.

## v0.1.6 — Devolución del director

Registra el primer checkpoint con el director (15/11/2026). No cambia código.

### Agregado
- `CLAUDE.md` §10.7: las sugerencias del director y cómo se incorporan al plan:
  - justificar filtros y procesamiento con teoría y experimentos medidos;
  - aumento de datos que afecte a video y señal juntos;
  - sobremuestreo por HR, promedio en inferencia y ensamble de semillas.

### Cambiado
- `CLAUDE.md` §10.5: decisión de pasar subject37 de validación a train. El cambio en `configs/splits.yaml` se aplica en una rama posterior, junto con la nueva corrida de referencia; test no cambia.
- `CLAUDE.md` §8: el corrimiento de cronograma (R3) se actualiza a 3 o 4 semanas, conversado con el director y dentro del margen previo a la memoria final.

## v0.1.5 — Notebook de PhysNet ejecutada

### Agregado
- Notebook `03_train_physnet_colab` ejecutada en Colab (GPU T4) con el código de la v0.1.4 (`9ffbe7d`), sobre la corrida `physnet_v2`.
  - Retoma la corrida ya entrenada con la v0.1.3 y, como ya había cortado por *early stopping*, no vuelve a entrenar. El aviso de cambio de código al retomar es esperable: el modelo sigue registrado con el commit `5a477a1`.
  - La evaluación en validación reproduce los valores de la v0.1.4 (MAE 5,34 bpm, Pearson 0,68).
  - Las celdas del gráfico y del entrenamiento se ejecutaron en orden inverso (contadores 15 y 16). El gráfico no depende de esa ejecución, porque lee `history.csv`, que no cambia al retomar.

## v0.1.4 — PhysNet: ablación del aumento por velocidad

Evidencia de la mitigación de R1 y R4 incorporada en la v0.1.3, y una corrección al retomar en GPU.

### Resultados
Dos corridas en Colab (GPU T4) con el código de la v0.1.3 (`5a477a1`, sin cambios sin commitear), idénticas salvo por `augment.speed_p`. Ambas cortaron por *early stopping* y eligieron la época 5 por `val_loss`. Se evalúa en validación; el test sigue sin usarse.

| Sujeto (HR aprox.) | MAE con aumento (`physnet_v2`) | MAE sin aumento (`physnet_v2_nospeed`) |
|---|---|---|
| subject8 (~110 bpm) | 0,81 | 0,39 |
| subject12 (~66 bpm) | 0,55 | 1,18 |
| subject27 (~72 a 99 bpm) | 1,17 | 1,33 |
| subject37 (~54 a 69 bpm) | **22,96** | **37,48** |
| subject44 (~75 a 90 bpm) | 0,63 | 0,58 |
| **Todos** (287 ventanas) | **5,34** (RMSE 16,14; r 0,68; `val_loss` 0,311) | **8,39** (RMSE 20,49; r 0,49; `val_loss` 0,343) |

MAE en bpm, ventanas de 10 s con paso de 1 s (`configs/eval.yaml`).

### Interpretación
- La diferencia la explica subject37, el único sujeto de validación con HR por debajo del rango de train (77 a 129 bpm). Con aumento, cerca de 24 de sus 59 ventanas caen en el segundo armónico (~110 bpm cuando la referencia es ~55); sin aumento, cerca de 40. Sin aumento falla casi siempre que la HR baja de ~60 bpm; con aumento sigue a la referencia en varios tramos de ~55 bpm.
- En los otros cuatro sujetos, que están dentro del rango de train, no hay diferencia apreciable (MAE medio de 0,79 vs. 0,87 bpm): el aumento no perjudica.
- La mitigación es **parcial**: subject37 sigue con un MAE de 23 bpm.
- Limitaciones: una sola semilla por configuración, y el efecto depende de un único sujeto de validación. subject23 (~61 bpm, en test) permitirá confirmarlo en la evaluación final.

### Agregado
- `CLAUDE.md` §8: evidencia de la ablación en la tabla de mitigaciones.

### Corregido
- Retomar una corrida en GPU fallaba con `TypeError: RNG state must be a torch.ByteTensor`. `last.pt` se cargaba directamente en CUDA, y los estados de los generadores aleatorios tienen que estar en CPU. Ahora se carga en CPU, y modelo y optimizador pasan a la GPU al restaurarse. Los tests corren en CPU y por eso no lo detectaban.

## v0.1.3 — PhysNet: aumento por velocidad, selección por pérdida y early stopping

Primera iteración de la tarea 3.4 (hiperparámetros). Ataca los dos problemas de la corrida exploratoria `physnet_v1`: el sobreajuste y la estimación en el segundo armónico para HR bajas.

### Agregado
- **Aumento de datos por velocidad temporal** (mitigación de R1 y R4):
  - reproduce cada tramo entre 0,6 y 1,4 veces su velocidad, video y PPG juntos, con probabilidad 0,5;
  - motivo: en train no hay ningún sujeto con HR menor a 77 bpm (rango 77 a 129), pero en val está subject37 (56 bpm) y en test subject23 (61 bpm). Con este rango, el modelo también ve ~46 a ~180 bpm;
  - si el tramo de origen no entra en el video o tiene frames sin cara, se usa el tramo original.
- **Pérdida de validación** (`val_loss`): 1 − Pearson de la señal sobre todos los tramos de validación. No depende de la estimación espectral, así que no la afectan los errores de armónico.
- **Criterio de selección configurable** (`train.select_by`), por defecto `val_loss`. En `physnet_v1`, el MAE de 5 sujetos eligió un pico aislado.
- ***Early stopping*** (`train.early_stopping_patience`, por defecto 5 épocas). Se respeta también al retomar.
- La evaluación agrega la columna `signal_loss` por sujeto.
- Notebook de Colab: gráfico de pérdida de train y val, mejor época según el criterio, e instrucciones para la corrida de ablación sin aumento por velocidad.
- `CLAUDE.md` §8: tabla con el estado de cada mitigación de riesgo, versión y evidencia.

### Cambiado
- `configs/physnet.yaml`: `lr` de 1e-3 a 1e-4, porque con 1e-3 sobreajustaba desde la época 2. Nombre de corrida `physnet_v2`.

### Verificado
- La HR de la etiqueta y la del video escalan juntas con el factor de velocidad: tramos de 90 bpm dan 54 bpm a 0,6x y 126 bpm a 1,4x.
- *Early stopping* sobre datos reales en CPU, también al retomar una corrida ya cortada.

## v0.1.2 — PhysNet: modelo, entrenamiento, evaluación y Colab

Infraestructura completa del enfoque de DL (Etapa 2). Todavía no hay un modelo entrenado: el entrenamiento completo se hace en Colab a partir de esta versión.

### Agregado
- `rppg.dl.physnet`: PhysNet (Yu et al., BMVC 2019), tomado del repositorio original del autor (MIT).
  - Verificado contra el código original: mismos 768.577 parámetros y salida idéntica con los mismos pesos.
- `rppg.dl.losses`: 1 − Pearson (vectorizada), la pérdida de PhysNet.
- `rppg.dl.dataset`: tramos de 128 frames de los recortes ya extraídos, con la PPG del oxímetro como etiqueta, estandarizados por tramo.
  - Aumento de datos (R1): espejado, deriva lenta de iluminación y ruido de sensor.
- `rppg.dl.train`: entrenamiento con checkpoints reanudables por época (R2).
  - Guarda modelo, optimizador, AMP, estado de los generadores aleatorios e historial.
  - Selecciona el mejor modelo por el MAE de HR en validación.
- `rppg.dl.infer`: inferencia sobre el video completo, por tramos.
- `rppg.eval.protocol` y `configs/eval.yaml`: **protocolo de evaluación común** a ambos enfoques (R5). Cualquier señal de pulso pasa por el mismo post-procesamiento que la PPG de referencia.
- `rppg.eval.splits`, `scripts/make_splits.py` y `configs/splits.yaml`: particiones fijas por sujeto (28 train, 5 val, 9 test, semilla 2026), con chequeo automático de que no compartan sujetos.
- Scripts: `train_dl.py` (con `--smoke` y `--resume`), `evaluate_dl.py` y `pack_for_colab.py`.
- `configs/physnet.yaml`: hiperparámetros iniciales. Se pueden pisar con `--set clave=valor`.
- Notebook `03_train_physnet_colab`: entrenamiento en la GPU gratuita de Colab.
  - Clona el repo privado con un token guardado en los secretos de Colab.
  - Guarda los checkpoints en Drive para retomar si se corta la sesión.
  - El test queda bloqueado hasta el final.
- `rppg.utils.provenance`: **asociación modelo–commit**.
  - Cada checkpoint guarda el commit, la rama, si había cambios sin commitear y el entorno (Python, torch, GPU).
  - `run_info.json` registra cada sesión de una corrida, y `history.csv` lleva el commit de cada época.
  - Si se retoma con código distinto, o con cambios sin commitear, el entrenamiento lo advierte.
  - `evaluate_dl.py` guarda en `<split>_eval_info.json` el commit del modelo y el de la evaluación.
- Tests: arquitectura, pérdida, estandarización, tramos, aumento, inferencia, entrenamiento con reanudación, procedencia, particiones y protocolo.
- CI: instala PyTorch para CPU para correr los tests del modelo.

### Corregido
- `--set` con notación científica sin punto (`train.lr=1e-4`) se interpretaba como texto (PyYAML). Ahora se convierte a número.
- Notebook de Colab:
  - clona de cero en cada ejecución y directamente en `REPO_REF`;
  - instala con `--ignore-requires-python`, porque Colab usa Python 3.13 y el límite `<3.13` es solo por mediapipe, que no corre en Colab;
  - agrega `src/` a la ruta de imports del kernel y el `import os` que faltaba;
  - verifica que `REPO_REF` tenga el código de DL.

### Decisiones
- 10.4: PhysNet, tomado del repositorio original (MIT) en lugar de rPPG-Toolbox (RAIL), para no heredar las restricciones de esa licencia.
- 10.5: test fijo de 9 sujetos y train/val con el resto, sin validación cruzada. El test se evalúa una sola vez, al cerrar la búsqueda de hiperparámetros.
- Recortes de 72×72 para la red (los guardados son de 128×128): ~3 veces menos costo por paso en la GPU gratuita (R2). Se revisa en la tarea 3.4.

### Verificado
- Prueba de humo en CPU con datos reales (3 sujetos de train, 2 de val, 3 épocas, ~1 min): la pérdida baja de 1,00 a 0,94 y la cadena completa funciona de punta a punta. No es un resultado del modelo.
- Reanudación exacta: 3 épocas entrenadas en dos sesiones dan los mismos valores que 3 épocas seguidas.
- Corrida completa en Colab (GPU T4, 30 épocas, ~1,5 min por época), con el commit `2b10429`.
  - Es una corrida exploratoria, no el resultado final.
  - Mejor época en validación: la 13 (MAE 7,8 bpm). Es un pico aislado: en el resto de las épocas el MAE está entre 11 y 12 bpm.
  - Hay sobreajuste: la pérdida de entrenamiento baja sin parar mientras el MAE de validación empeora desde la época 2.
  - subject37 (HR de ~55 bpm) se estima en el segundo armónico (~2× la HR). Los otros 4 sujetos de validación quedan con MAE entre 0,4 y 1,7 bpm.
  - Esto se aborda en la tarea 3.4 (hiperparámetros).

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

