# Contexto del proyecto: Análisis de signos vitales mediante visión por computadora

> Guardar este archivo como `CLAUDE.md` en la raíz del repositorio. Resume el Plan de Trabajo aprobado (FIUBA, curso Gestión de Proyectos, 24/08/2026 al 18/10/2026) y las decisiones técnicas discutidas. El plan es la fuente de verdad sobre alcance, requerimientos y cronograma. Lo que figura en "Decisiones abiertas" (sección 10) todavía no está resuelto.

## 1. Resumen

Trabajo Final de la Carrera de Especialización en Inteligencia Artificial (FIUBA, UBA).

- **Título:** Análisis de signos vitales mediante técnicas de visión por computadora
- **Autor:** Ing. Santiago Ruiz
- **Director:** Esp. Ing. Gerardo Alexis Vilcamiza Espinoza
- **Inicio:** 24/08/2026. **Defensa pública:** abril de 2027.
- **Dedicación estimada:** 600 h (modelada como 4 h/día, lunes a viernes).
- **Naturaleza:** prueba de concepto técnica, individual, sin financiamiento externo. No es un dispositivo médico.

**Problema.** Medir signos vitales requiere sensores en contacto con la piel (oxímetro, electrodos, banda pectoral), lo cual es inviable en una consulta de telemedicina. La fotopletismografía remota (rPPG) estima signos vitales a partir de video, usando las variaciones sutiles de color de la piel que produce el flujo sanguíneo.

**Propuesta.** Estimar **frecuencia cardíaca (HR)** y, a partir de la misma señal, **frecuencia respiratoria (RR)** desde video facial. Se comparan un enfoque **clásico** (procesamiento de señales) y uno de **deep learning** (red espacio-temporal) sobre datasets públicos, contra mediciones de referencia de sensores de contacto.

**Cliente (simbólico):** médicos de atención remota, en el contexto de prepagas con telemedicina. **Usuario final:** profesional de la salud y paciente.

## 2. Autor y entorno de trabajo

- Ingeniero electrónico (UNLaM), cursando la Especialización en IA (FIUBA). Cómodo con Python, OpenCV, YOLO, PyTorch y consideraciones de despliegue en edge. Tiene experiencia previa en sistemas embebidos.
- **Notebook sin GPU dedicada, 20 GB de RAM.** Los entrenamientos locales anteriores le resultaron lentos.
- **Estrategia de cómputo (híbrida):**
  - Desarrollo, debugging y pipeline clásico: local, con un subconjunto chico (2 o 3 videos).
  - Entrenamiento completo del modelo de DL: Google Colab con GPU (nivel gratuito, T4). No hay presupuesto para Colab Pro.
  - Usar checkpoints periódicos para poder reanudar ante cortes de sesión (riesgo R2).
- **Idioma:** conversar y documentar en español.
  - **Comentarios, docstrings y celdas markdown de las notebooks: en español** (13/10/2026), porque el posgrado es de habla hispana.
  - Mensajes de commit y `RELEASE_NOTES.md`: en español.
  - Identificadores de código y mensajes de excepción: en inglés.
- **Versionado:**
  - Cada PR a `main` lleva `#major`, `#minor` o `#patch` en el título y agrega una sección al principio de `RELEASE_NOTES.md`.
  - Tags y releases los crean los workflows compartidos de `santiagoruizvt/sr-workflows`. No crear tags a mano ni editar `.github/workflows/pre-release.yml` ni `release.yml`.
  - Una rama por tarea, integrada por PR. El PR que completa un hito lleva `#minor`; los intermedios, `#patch`.

## 3. Alcance

**Incluye:**
1. Relevamiento del estado del arte en rPPG (clásico a deep learning).
2. Pipeline clásico (baseline): detección facial, ROI, extracción de señal de color, filtrado, estimación espectral de HR.
3. Modelo de deep learning espacio-temporal (tipo CNN 3D) para estimar HR desde video.
4. Entrenamiento y validación sobre datasets públicos (UBFC-rPPG principal, PURE complementario).
5. Comparación clásico vs. DL con MAE, RMSE y correlación de Pearson.
6. Documentación de resultados, limitaciones y condiciones de uso válidas.

**No incluye:** presión arterial ni SpO2; interfaz de usuario ni integración con una plataforma de telemedicina; validación clínica ni aprobación regulatoria; despliegue en producción ni tiempo real sobre hardware embebido.

**Supuestos:** datasets públicos disponibles durante todo el proyecto; recursos de cómputo (GPU si hace falta) suficientes; condiciones de captura razonablemente estables (iluminación, rostro visible, sin oclusiones) representativas de una videollamada.

## 4. Enfoque técnico

### 4.1 Arquitectura del sistema (diagrama en bloques)

Video facial → detección facial y selección de ROI → **una única señal rPPG** (pipeline clásico o red de DL) → filtrado en dos bandas → estimación de HR y de RR → comparación contra sensor de referencia.

### 4.2 Pipeline clásico (Etapa 1, baseline)

1. Detección y seguimiento facial (MediaPipe Face Mesh u otro detector liviano).
2. ROI de piel confiable (frente y mejillas; evitar ojos, boca, vello).
3. Señal promedio de color por frame sobre la ROI. Canal verde como baseline simple. Considerar CHROM (de Haan & Jeanne, 2013) o POS (Wang et al., 2017) para robustez ante movimiento e iluminación.
4. Filtro pasa-banda: **HR ≈ 0,7 a 4 Hz** (42 a 240 bpm).
5. **RR ≈ 0,1 a 0,5 Hz** (6 a 30 respiraciones/min), filtrando la misma señal con otra banda.
6. Estimación por análisis espectral (FFT o Welch): el pico dominante en cada banda da HR y RR.

### 4.3 Modelo de deep learning (Etapa 2)

- Punto de partida realista: CNN espacio-temporal 3D tipo **PhysNet** (Yu et al., 2019). Alternativa: DeepPhys (Chen & McDuff, 2018). Transformers (PhysFormer) quedan fuera por costo de datos y cómputo.
- Pérdida orientada a la señal temporal (correlación negativa de Pearson contra la señal de referencia), eventualmente combinada con error sobre HR.
- Aumento de datos (brillo, contraste, ruido) para mitigar el riesgo de falta de generalización (R1).
- Considerar partir de **rPPG-Toolbox** (Liu et al., NeurIPS 2023, repositorio abierto de la comunidad) como base de referencia en lugar de reimplementar todo.
- Dataset chico (decenas de videos): entrenamientos cortos. En la literatura PhysNet sobre UBFC-rPPG converge en minutos a pocas horas de GPU de gama media.

### 4.4 Cómo se obtiene la RR

Una sola señal rPPG; se filtra en dos bandas. La respiración modula la señal cardíaca (amplitud, frecuencia, línea de base). No se usa ROI de tórax ni flujo óptico, porque en videollamada el encuadre suele ser solo la cara.

## 5. Datos

- **UBFC-rPPG** (dataset principal), DATASET_2, en `data/raw/subjectN/`. Verificado sobre los archivos (13/10/2026):
  - 42 videos de **~52 a 68 s**, 640x480, sin compresión.
  - fps irregular: entre ~23 y ~30, con huecos de hasta ~1 s.
  - `ground_truth.txt` trae PPG del oxímetro, HR del oxímetro (con cortes, no usar) y timestamps.
  - Sin señal respiratoria.
  - Detalle en `docs/arquitectura_pipeline_clasico.md` §3.1.
- **PURE** (complementario, opcional).
- **Particiones:** train/val/test **por sujeto** (sin sujetos compartidos entre conjuntos). Es un requerimiento explícito.
- Usar solo datasets con consentimiento informado y licencia académica. Revisar los términos de uso antes de descargar.

## 6. Métricas y requerimientos (resumen de la sección 6 del plan)

**Métricas:** MAE, RMSE y correlación de Pearson entre la HR estimada y la de referencia.

| Categoría | Requerimientos principales |
|---|---|
| Funcionales | Detectar rostro en ≥95 % de los frames con iluminación estable; extraer señal rPPG de la ROI; estimar HR (bpm); estimar RR (resp/min) desde la misma señal; pipeline clásico como baseline; al menos un modelo de DL espacio-temporal; procesamiento offline de videos de los datasets. Opcional (baja): cámara web en vivo. |
| Desempeño | MAE de HR ≤ 5 bpm sobre UBFC-rPPG (ambos enfoques); Pearson > 0,8; procesar 1 min de video en ≤ 2 min (prioridad baja). |
| Datos y validación | Evaluar al menos en UBFC-rPPG; PURE si hay tiempo; train y eval disjuntos por sujeto. |
| Documentación | Arquitecturas de ambos enfoques; métricas y condiciones de obtención; limitaciones y condiciones de uso válidas. |
| Regulatorios y éticos | Solo datasets públicos con consentimiento y licencia académica; no presentar el sistema como dispositivo médico certificado. |

Referencia de la literatura (orden de magnitud, no objetivo): PhysNet reporta MAE ≈ 3 bpm y Pearson ≈ 0,97 sobre UBFC-rPPG. Los umbrales del plan (5 bpm, 0,8) son un piso conservador.

## 7. Plan de trabajo

### 7.1 WBS (600 h)

| Grupo | Subtareas (horas) |
|---|---|
| 1. Relevamiento y diseño (60) | 1.1 Estado del arte (20); 1.2 Selección y descarga de datasets (10); 1.3 Arquitectura del pipeline clásico y del modelo de DL (20); 1.4 Métricas de evaluación (10) |
| 2. Pipeline clásico (120) | 2.1 Detección facial y ROI (30); 2.2 Extracción de señal rPPG (30); 2.3 Filtrado y estimación de HR (30); 2.4 Filtrado y estimación de RR (30) |
| 3. Modelo de DL (180) | 3.1 Preprocesamiento (30); 3.2 Arquitectura (40); 3.3 Entrenamiento (40); 3.4 Hiperparámetros (40); 3.5 Validación sobre test (30) |
| 4. Evaluación (100) | 4.1 MAE/RMSE (30); 4.2 Pearson (20); 4.3 Análisis comparativo (30); 4.4 Limitaciones y condiciones de uso (20) |
| 5. Documentación y cierre (140) | 5.1 Memoria v1 (30); 5.2 Revisión del director (40); 5.3 Memoria final (30); 5.4 Presentación de defensa (20); 5.5 Ensayo de defensa (20) |

### 7.2 Dependencias (AoN)

- 1.1 y 1.2 sin predecesores (en paralelo). 1.3 ← {1.1, 1.2}. 1.4 ← {1.1}.
- 2.1 ← 1.3. 2.2 ← 2.1. 2.3 ← 2.2. 2.4 ← 2.2 (2.3 y 2.4 en paralelo).
- 3.1 ← {1.2, 1.3}. 3.2 ← 3.1. 3.3 ← 3.2. 3.4 ← 3.3. 3.5 ← {3.4, 1.4}.
- 4.1 y 4.2 ← {2.3, 2.4, 3.5, 1.4} (en paralelo). 4.3 ← {4.1, 4.2}. 4.4 ← 4.3.
- 5.1 ← 4.4. 5.2 ← 5.1. 5.3 ← 5.2. 5.4 ← 5.1. 5.5 ← {5.3, 5.4}.

**Camino crítico:** 1.1 → 1.3 → 3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 4.1 → 4.3 → 4.4 → 5.1 → 5.2 → 5.3 → 5.5.
La rama de DL gobierna el cronograma. El pipeline clásico tiene holgura grande (~22 días hábiles). Mientras el modelo entrena en la GPU se puede avanzar con el pipeline clásico. Semicríticas: 1.2 (holgura baja) y 4.2.

### 7.3 Cronograma (aprobado)

| Grupo | Fechas |
|---|---|
| 1. Relevamiento y diseño | 24/08 al 04/09/2026 |
| 2. Pipeline clásico | 07/09 al 07/10/2026 |
| 3. Modelo de DL | 07/09 al 06/11/2026 |
| 4. Evaluación | 09/11 al 04/12/2026 |
| 5. Documentación y cierre | 07/12/2026 al 10/04/2027 |
| Hito: entrega de memoria final | 15/03/2027 |
| Hito: defensa pública | 15/04/2027 |

El Gantt aprobado deja margen entre el fin del trabajo técnico y la defensa. Es contingencia y tiempo administrativo, no tiempo muerto.

### 7.4 Presupuesto

USD 10.390 (600 h a USD 17/h = 10.200 directos + 190 indirectos). Equivale a ~ARS 16.052.550 al dólar oficial BNA de $1.545 (25/09/2026).

## 8. Riesgos (RPN = S×O; mitigación obligatoria si RPN ≥ 35)

| Riesgo | S | O | RPN | RPN* |
|---|---|---|---|---|
| R1. El modelo de DL no generaliza fuera de laboratorio | 7 | 6 | 42 | 21 |
| R2. Indisponibilidad de GPU en la nube | 6 | 6 | 36 | 18 |
| R3. Atraso de cronograma (único responsable) | 8 | 5 | 40 | 24 |
| R4. Calidad/representatividad de los datasets | 6 | 4 | 24 | - |
| R5. Comparación no homogénea entre enfoques | 5 | 4 | 20 | - |

Mitigaciones: R1 aumento de datos + documentar condiciones de uso válidas + (si se puede) pequeña captura propia en otras condiciones. R2 checkpoints periódicos, reducir batch o modelo si hace falta. R3 hitos semanales, priorizar camino crítico, margen de contingencia. R5 se evita fijando desde el inicio un protocolo de evaluación común (mismos splits y métricas para ambos enfoques).

## 9. Calidad (verificación = caja blanca; validación = con el director)

Doce requerimientos críticos tienen verificación y validación definidas. Lo que debe quedar automatizado en el código:
- % de frames con detección facial exitosa.
- MAE, RMSE y Pearson sobre el conjunto de evaluación completo, para ambos enfoques.
- Chequeo automático de que no haya sujetos compartidos entre train y test.
- Pruebas unitarias por etapa del pipeline clásico (detección, ROI, filtrado, estimación espectral).
- Pipeline reproducible de punta a punta con un comando o script único.

## 10. Decisiones abiertas y puntos a verificar

1. **Ground truth de RR.** El plan dice "contra una referencia disponible", pero hasta donde se sabe UBFC-rPPG provee señal PPG y HR de oxímetro, **no una señal respiratoria**. Verificar qué trae cada dataset. Opciones: derivar una RR de referencia desde la modulación respiratoria del PPG del oxímetro, usar un dataset con cinturón respiratorio, o tratar la RR como resultado exploratorio sin métrica formal. Decidirlo temprano porque afecta las tareas 2.4 y 4.1/4.2. Si cambia, hay que reflejarlo en el plan. **Actualización (13/10/2026):**
   - Confirmado que UBFC-rPPG no trae señal respiratoria.
   - La RR derivada de la PPG del oxímetro es inconsistente entre RIIV, RIAV y RIFV en ~75 % de los sujetos.
   - Opciones y evidencia en `docs/arquitectura_pipeline_clasico.md` §4.1–4.3.
   - **Resuelto (13/10/2026): (c) + (d).** RR exploratoria sobre UBFC-rPPG y métrica formal contra cinturón respiratorio en COHFACE (alternativa: MAHNOB-HCI), sujeto a obtener la licencia. La EULA de COHFACE requiere un firmante con cargo permanente (director o facultad).
   - **Actualización (25/10/2026): COHFACE queda postergado.**
     - Por ahora solo (d): la RR se estima sobre UBFC-rPPG aunque no haya referencia respiratoria.
     - Se reporta como resultado exploratorio: distribución, consistencia entre RIIV, RIAV y RIFV, y comparación con las mismas vías calculadas sobre la PPG del oxímetro.
     - No se reportan métricas formales de RR.
2. ~~**Subconjunto de UBFC-rPPG.**~~ **Resuelto:** DATASET_2, 42 sujetos (subject1 a subject49, no consecutivos).
3. ~~**Detector facial.**~~ **Resuelto (13/10/2026):** MediaPipe Face Landmarker. Ver `docs/arquitectura_pipeline_clasico.md` §8, junto con D2 (fuente de RR) y D3 (ventanas de evaluación).
4. ~~**Arquitectura de DL concreta.**~~ **Resuelto (25/10/2026): PhysNet**, la misma arquitectura que usa rPPG-Toolbox.
   - Se toma del repositorio original del autor (ZitongYu/PhysNet, licencia MIT) y no de rPPG-Toolbox. rPPG-Toolbox está bajo licencia RAIL, que restringe usos en salud (p. ej. "diagnosticar sin supervisión humana" o fijar primas de seguros) y obliga a trasladar esas restricciones.
   - La implementación (`rppg.dl.physnet`) se verificó contra el código original: mismos parámetros (768.577) y salida idéntica con los mismos pesos.
   - Entrenamiento y evaluación propios, con el protocolo común.
5. ~~**Esquema de particiones.**~~ **Resuelto (25/10/2026):** test fijo + train/val, sin validación cruzada.
   - `configs/splits.yaml`, generado con `scripts/make_splits.py`, semilla 2026: 28 sujetos de train, 5 de val y 9 de test.
   - No regenerarlo una vez que haya resultados.
   - El test se evalúa una sola vez, al cerrar la tarea 3.4. El clásico se reporta sobre el mismo test y sobre los 42 sujetos.
6. **Captura propia opcional** (webcam + oxímetro) para evaluar fuera de los datasets: depende de tiempo disponible.

## 11. Stack y estructura sugerida del repo

- Python, PyTorch, OpenCV, MediaPipe, NumPy/SciPy (filtros y FFT), pandas, Jupyter para exploración y scripts para entrenamiento final.
- Todo open source (sin costos de software).

```
.
├── CLAUDE.md
├── README.md
├── pyproject.toml / requirements.txt
├── configs/            # .yaml: rutas, splits, hiperparámetros
├── data/               # NO versionar datos (raw/, processed/)
├── src/
│   ├── face/           # detección facial y ROI
│   ├── classical/      # extracción rPPG, filtros, estimación espectral
│   ├── dl/             # dataset, modelo, entrenamiento, inferencia
│   ├── eval/           # MAE, RMSE, Pearson, comparación
│   └── utils/
├── notebooks/          # exploración
├── scripts/            # ej.: run_classical.py, train_dl.py, evaluate.py
├── tests/              # unitarias por etapa
└── docs/               # decisiones, resultados, borradores de la memoria
```

## 12. Primeros pasos sugeridos

1. Crear el repo, el entorno y la estructura anterior. Dejar los datos fuera de git.
2. Revisar la documentación de UBFC-rPPG y resolver el punto 10.1 (ground truth de RR).
3. Descargar UBFC-rPPG y escribir un loader que lea video y ground truth. Probar con 2 o 3 sujetos.
4. Implementar el baseline: detección facial → ROI → señal de canal verde → pasa-banda → FFT → HR. Medir MAE sobre pocos videos.
5. Agregar CHROM/POS y comparar contra el baseline.
6. Definir splits por sujeto y el protocolo de evaluación común (misma tabla de métricas para ambos enfoques) antes de tocar el modelo de DL.
7. Armar el modelo de DL localmente con un subconjunto mínimo para validar que entrena. Mover el entrenamiento completo a Colab con checkpoints.

## 13. Referencias

- Poh, McDuff & Picard (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. *Optics Express*.
- de Haan & Jeanne (2013). Robust pulse rate from chrominance-based rPPG (CHROM). *IEEE TBME*.
- Wang et al. (2017). Algorithmic principles of remote PPG (POS). *IEEE TBME*.
- Chen & McDuff (2018). DeepPhys: video-based physiological measurement using convolutional attention networks. *ECCV*.
- Yu, Li & Zhao (2019). Remote photoplethysmograph signal measurement from facial videos using spatio-temporal networks (PhysNet). *BMVC*.
- Bobbia et al. (2019). Unsupervised skin tissue segmentation for remote photoplethysmography (dataset UBFC-rPPG). *Pattern Recognition Letters*.
- Liu et al. (2023). rPPG-Toolbox: deep remote PPG toolbox. *NeurIPS Datasets and Benchmarks*.
- Khanam, Al-Naji & Chahl (2019). Remote monitoring of vital signs in diverse non-clinical and clinical scenarios using computer vision systems: a review. *Applied Sciences* 9, 4474.
- Sakib et al. (2025). A state-of-the-art survey of remote photoplethysmography for contactless health parameters sensing. *WIREs Data Mining and Knowledge Discovery*.

## 14. Cómo trabajar conmigo en este repo

- Antes de implementar una etapa, confirmar contra este documento (alcance y requerimientos) y avisar si algo del plan debería cambiar.
- Ante una decisión abierta (sección 10), plantear las opciones con sus trade-offs y esperar mi elección.
- Priorizar siempre lo que está en el camino crítico (rama de DL) y mantener reproducibilidad: semillas fijas, configs en `.yaml`, nada hardcodeado.
- No inventar resultados ni números de la literatura. Si cito un valor, indicar de dónde sale.