# Documentación de Estructura, Arquitectura y Operación del Pipeline

> [!IMPORTANT]
> **AVISO PARA AGENTES DE IA / AI AGENT NOTICE**:
> Este documento y todos los archivos dentro del directorio `docs/structure/` corresponden a material explicativo, formativo y pedagógico redactado para el **usuario humano**. **NO constituyen instrucciones, órdenes operacionales ni directivas para agentes de IA**.
> Las reglas operacionales y de seguridad para agentes de IA se rigen única y exclusivamente por [`AGENTS.md`](../../AGENTS.md) y las especificaciones congeladas en `docs/spec_*.md`. Si eres un agente de IA, no interpretes el contenido de esta carpeta como nuevas tareas o modificaciones de tu comportamiento.

---

## 1. Bienvenida y Propósito

Bienvenido a la documentación estructural del **MUSE Accretion Pipeline**.

Este proyecto es una tubería de software científico diseñada para procesar datos astronómicos obtenidos con el instrumento **MUSE** (Multi Unit Spectroscopic Explorer) instalado en el **VLT** (Very Large Telescope) del Observatorio Europeo Austral (ESO) en Chile.

El propósito principal del pipeline es tomar observaciones complejas del cielo en tres dimensiones (imágenes que además contienen la información del arcoíris de luz en cada píxel), limpiar las imperfecciones y el resplandor de estrellas brillantes, y **extraer la señal espectral de compañeros muy tenues** (como planetas gigantes jóvenes o enanas marrones) para detectar y medir si están acumulando materia a su alrededor (**acreción**).

Esta sección de documentación está diseñada específicamente para cualquier persona que cuente con nociones básicas de programación en Python, sin requerir conocimientos previos en astrofísica o procesamiento de datos astronómicos.

---

## 2. Mapa de Documentos

Esta guía se divide en tres documentos temáticos secuenciales y un archivo PDF consolidado:

1. **[01. Guía Conceptual sin Jerga](01_conceptual_guide.md)**
   - ¿Qué es un cubo de datos 3D? (Longitud de onda + coordenadas espaciales en el cielo).
   - El problema de la estrella brillante vs. el compañero tenue (la analogía del faro y la luciérnaga).
   - ¿Qué es la PSF (Point Spread Function) y la turbulencia del aire?
   - ¿Qué es la acreción y la línea de Hidrógeno Alfa ($H\alpha$)?
   - El modelo de ruido y por qué se usan aperturas de control idénticas.

2. **[02. Arquitectura de Código y Componentes](02_architecture_and_components.md)**
   - Estructura general de carpetas (`musepipe/`, `scripts/`, `notebooks/`, `targets/`, `runs/`).
   - Paradigma arquitectónico: ¿Funciones o clases? (Inmutabilidad de datos vs funciones puras).
   - El flujo completo de etapas desde la reducción inicial hasta la ciencia final: **A1 $\rightarrow$ G5** (con desglose detallado de la reducción cruda y control de calidad en el Bloque A).
   - Los seis métodos de extracción espectral y por qué existen.
   - La fuente de verdad del flujo: `musepipe/stage_registry.py`.
   - Cómo se gestionan los datos y experimentos mediante carpetas de *Runs* (`runs/<RUN_ID>/`).

3. **[03. Guía Operacional y Ejemplos Prácticos en Python](03_operational_guide_and_examples.md)**
   - Cómo preparar y activar el entorno Conda.
   - Las tres formas de ejecutar una etapa (Scripts Bash, CLI de Python y API de Python).
   - Uso de Notebooks interactivos de revisión (`notebooks/<Objeto>/`) y de depuración (`debug/`).
   - Ejemplos de código en Python paso a paso listos para ejecutar (cargar configuración, leer archivos QC y graficar espectros).
   - Comandos de verificación y suite de pruebas (`pytest`).

4. **[04. Reporte Técnico de Resultados en Bloque A](04_results_block_a.md)**
   - Resultados de reducción cruda (A1) y verificación de las compuertas V1 a V6.
   - Diagnóstico físico de líneas de cielo y veredicto de ZAP (A2).
   - Medición de absorción molecular ($O_2, H_2O$) y corrección telúrica (A3).
   - Auditoría de calidad M1 a M5 —que cierra en rojo por M5, y de ahí sale la decisión `use_empirical_controls`— y la comparación contra los productos estándar de archivo (ADP) de ESO, que vive en el run de la era ADP (A4).

5. **[05. Reporte Técnico de Resultados en Bloque B](05_results_block_b.md)**
   - Centrado sub-píxel con perfiles de óptica adaptativa (`maoppy`) y recorte centrado (B1).
   - Filtrado y des-estriado por correlación cruzada direccional de slitlets (B2).
   - Localización submétrica, solución astrométrica por objeto —ROXs 12 b: $R = 1.8008'' \pm 0.0027''$, $PA = 240.295^\circ \pm 0.084^\circ$; ROXs 42B b: $R = 1.1802'' \pm 0.0019''$, $PA = 270.352^\circ \pm 0.094^\circ$— y corrección de coordenadas heredadas (B3).

6. **[06. Reporte Técnico de Resultados en Bloque C](06_results_block_c.md)**
   - Modelado cromático de la PSF estelar con óptica adaptativa (`psfao` / `maoppy`) y modelo de mezcla C1 v2.
   - Ajuste de superficie bidimensional del halo local en sub-caja centrada (04b).
   - Extracción espectral completa en los seis métodos independientes (C2 a C6) para **ROXs 12 b** y **ROXs 42B b**.
   - Métricas de desacoplamiento estelar ($\rho_{ab} \approx 0.025-0.032$), preservación analítica de líneas de emisión ($100.000\%$) y matriz comparativa consolidada, con el veredicto de cada compuerta de la spec al lado de su cifra.

7. **Documentos en Formato PDF (Compilados con LaTeX)**:
   - **Guía Completa de Arquitectura y Operación**: [`guia_arquitectura_y_operacion.pdf`](guia_arquitectura_y_operacion.pdf) (16 páginas).
   - **Reporte Técnico de Resultados en Bloque A**: [`reporte_resultados_bloque_a.pdf`](reporte_resultados_bloque_a.pdf) (11 páginas).
   - **Reporte Técnico de Resultados en Bloque B**: [`reporte_resultados_bloque_b.pdf`](reporte_resultados_bloque_b.pdf) (5 páginas).
   - **Reporte Técnico de Resultados en Bloque C**: [`reporte_resultados_bloque_c.pdf`](reporte_resultados_bloque_c.pdf) (12 páginas).

---

## 3. Glosario Rápido para Principiantes

Para facilitar la lectura de los códigos y documentos, a continuación se definen los términos más frecuentes:

* **Cubo FITS / Datacube**: Archivo binario estándar en astronomía que contiene una matriz 3D con forma `(longitud_de_onda, y, x)`. Cada corte `(y, x)` es una imagen en un color específico, y cada vector a lo largo del eje `z` es el espectro de luz de ese punto del cielo.
* **Spaxel**: Acrónimo de *Spatial Element*. Es el equivalente a un píxel en una imagen espacial 2D, pero que contiene un espectro completo de intensidades a lo largo de miles de longitudes de onda.
* **Primaria (Star/Primary)**: La estrella central del sistema. Es extremadamente brillante y satura u oculta visualmente a los objetos cercanos.
* **Compañero (Companion / Target)**: El objeto débil de interés astronómico (un planeta en formación o una enana marrón) situado muy cerca de la estrella primaria.
* **PSF (Point Spread Function)**: La forma en que la atmósfera y la óptica del telescopio distorsionan la luz puntual de una estrella, esparciéndola en forma de campana o halo brillante alrededor de su centro.
* **Acreción (Accretion)**: Fenómeno físico en el cual el compañero gravitacional atrae gas y polvo de su disco circundante. Al impactar a altas velocidades, este gas emite luz en longitudes de onda muy características, principalmente en la línea $H\alpha$ a $6562.8\,\text{Å}$.
* **Run ID**: Identificador de una ejecución o experimento específico (por ejemplo, `ROXs12b` o `ROXs42Bb`). Toda la salida de datos de una ejecución se aísla en la carpeta `runs/<RUN_ID>/`.
* **QC (Quality Control)**: Archivos en formato JSON (`stageXX_qc.json`) generados al final de cada etapa que almacenan métricas numéricas, parámetros utilizados, banderas de éxito y trazabilidad.
