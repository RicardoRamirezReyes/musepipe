# 03. Guía Operacional y Ejemplos Prácticos en Python

> [!IMPORTANT]
> **AVISO PARA AGENTES DE IA / AI AGENT NOTICE**:
> Este documento corresponde a material explicativo y formativo redactado para el **usuario humano**. **NO constituye instrucciones ni órdenes operacionales para agentes de IA**.
> Las directivas y reglas para agentes están gobernadas exclusivamente por [`AGENTS.md`](../../AGENTS.md).

---

## 1. Configuración del Entorno de Trabajo

Todo el software del pipeline requiere el entorno Conda denominado `MUSE`, el cual contiene Python 3.10 y las librerías científicas necesarias (`astropy`, `numpy`, `scipy`, `matplotlib`, `pytest`).

### Activación
Abre una terminal en la raíz del repositorio y ejecuta:

```bash
conda activate MUSE
```

Para verificar que el entorno y las librerías principales están listas:

```bash
python -c "import astropy, numpy, scipy, matplotlib; print('Entorno MUSE configurado correctamente.')"
```

---

## 2. Selección de un *Run* (Identificador de Ejecución)

El pipeline opera siempre sobre un experimento o dataset identificado por un `run_id` (por ejemplo: `ROXs12b`, `ROXs42Bb`).

Para indicar sobre qué run se desea trabajar, se puede usar:
1. **La variable de entorno `MUSE_RUN_ID`** (recomendado para sesiones de terminal):
   ```bash
   export MUSE_RUN_ID=ROXs12b
   ```
2. **El argumento `--run-id`** al invocar un script:
   ```bash
   python -m musepipe.stages.stage_x03_psffit --run-id ROXs12b
   ```

Para verificar rápidamente qué run está activo sin modificar ningún archivo:

```bash
MUSE_RUN_ID=ROXs12b python -c "from musepipe.config import load_run_config; print('Run activo:', load_run_config().run_id)"
```

---

## 3. Las Tres Formas de Ejecutar una Etapa

Supongamos que deseamos ejecutar la etapa **C4** (Ajuste simultáneo de PSF con `psffit`). Existen tres formas equivalentes de hacerlo:

### Forma A: Mediante Scripts de Consola (Bash)
La carpeta `scripts/` contiene lanzadores listos para cada etapa:

```bash
./scripts/stage_x03_psffit.sh --run-id ROXs12b
```

### Forma B: Mediante el Módulo de Python desde la Terminal
Cada etapa es un módulo ejecutable con soporte completo de argumentos (`--run-id`, `--help`):

```bash
python -m musepipe.stages.stage_x03_psffit --run-id ROXs12b
```

### Forma C: Mediante la API de Python (Scripts o Jupyter)
Si deseas integrar una etapa dentro de un flujo personalizado en Python:

```python
from musepipe.stages import run_stage_x03_psffit

# Ejecuta la etapa y retorna el diccionario de métricas de calidad (QC)
qc_resultado = run_stage_x03_psffit(run_id="ROXs12b")

print("Etapa completada con éxito.")
print("Archivo QC generado:", qc_resultado.get("output_qc_file"))
```

---

## 4. Uso de los Cuadernos Interactivos (Jupyter Notebooks)

El proyecto incluye dos tipos de cuadernos en `notebooks/`:

### 1. Notebooks de Revisión (`notebooks/<Objeto>/`)
Permiten auditar visualmente los resultados de cada etapa (gráficos de residuos, curvas de flujo, mapas 2D).
- Cada objeto tiene su propia carpeta: `notebooks/ROXs12b/`, `notebooks/ROXs42Bb/`.
- Cada etapa tiene un notebook dedicado: `A1_raw_reduction.ipynb` ... `G5_final_synthesis.ipynb`.
- **Regeneración automática**: Si se actualizan parámetros o el código de visualización, los notebooks se regeneran con:
  ```bash
  python scripts/build_review_notebooks.py --target ROXs12b
  ```

### 2. Notebooks de Depuración y Análisis (`notebooks/<Objeto>/debug/`)
Diseñados para experimentar con fórmulas matemáticas o cambiar parámetros sin alterar la cadena oficial:
- Se generan mediante:
  ```bash
  python scripts/build_debug_notebooks.py --target ROXs12b
  ```
- Contienen celdas de detección de cambios de código (*drift cells*) que avisan si la función en el cuaderno difiere de la librería `musepipe`.

---

## 5. Ejemplos Prácticos en Python (Paso a Paso)

A continuación se presentan ejemplos interactivos que puedes copiar y pegar en un archivo `.py` o en una sesión interactiva de Python.

### Ejemplo 1: Cargar la Configuración de un Run

```python
"""
Ejemplo 1: Inspeccionar la configuración de un Run.
"""
from musepipe.config import load_run_config

# Carga la configuración del run ROXs12b
run_cfg = load_run_config(run_id="ROXs12b")

print("=== Configuración del Run ===")
print("ID del Run:", run_cfg.run_id)
print("Ruta base:", run_cfg.paths.run_dir)

# Leer parámetros astronómicos configurados
companion_pos = run_cfg.config.get("target_position_xy", "No definido")
print("Posición estimada del compañero [x, y]:", companion_pos)
```

---

### Ejemplo 2: Leer e Interpretar un Archivo de Control de Calidad (QC)

Cada etapa guarda sus resultados en un archivo JSON estructurado:

```python
"""
Ejemplo 2: Leer las métricas de calidad de la etapa C4 (PSF Fitting).
"""
import json
from pathlib import Path
from musepipe.paths import RunPaths

# Obtener rutas estándar
paths = RunPaths(project_root=".", run_id="ROXs12b")
qc_file = paths.stage_dir / "spec_psffit_qc.json"

if qc_file.exists():
    with open(qc_file, "r") as f:
        qc_data = json.load(f)

    print("=== Métricas de Calidad de C4 (PSFFIT) ===")
    print("Estado de ejecución:", qc_data.get("status", "Desconocido"))
    print("Chi-cuadrado reducido promedio:", qc_data.get("mean_reduced_chi2", "N/A"))
    print("Separación medida (arcsec):", qc_data.get("separation_arcsec", "N/A"))
else:
    print(f"El archivo QC no existe aún en: {qc_file}")
```

---

### Ejemplo 3: Cargar y Graficar un Espectro FITS Calibrado

Una vez ejecutada la etapa **D2**, los espectros finales del compañero se encuentran calibrados en unidades físicas CGS. Podemos leerlos y visualizarlos fácilmente con `astropy` y `matplotlib`:

```python
"""
Ejemplo 3: Visualizar el espectro calibrado del compañero.
"""
import matplotlib.pyplot as plt
from astropy.io import fits
from musepipe.paths import RunPaths

# Localizar el espectro del método psffit
paths = RunPaths(project_root=".", run_id="ROXs12b")
spec_file = paths.stage_dir / "spec_calibrated_psffit_object.fits"

if spec_file.exists():
    with fits.open(spec_file) as hdul:
        # El HDU 1 contiene la tabla de datos espectrales
        data = hdul[1].data
        wavelength = data["WAVELENGTH"]  # Longitud de onda en Angstroms
        flux = data["FLUX"]              # Flujo en erg / (s cm^2 A)
        error = data["FLUX_ERR"]        # Barra de error empírica

    # Crear gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(wavelength, flux, label="Espectro Compañero (psffit)", color="navy", lw=1)
    plt.fill_between(wavelength, flux - error, flux + error, color="royalblue", alpha=0.3, label="Incertidumbre 1σ")

    # Marcar la posición esperada de H-alpha (6562.8 Å)
    plt.axvline(6562.8, color="red", linestyle="--", label="Línea Hα (6562.8 Å)")

    plt.xlabel("Longitud de Onda [Å]")
    plt.ylabel("Flujo [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    plt.title("Espectro Calibrado - ROXs 12 b")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Guardar figura
    plt.savefig("mi_primer_espectro.png", dpi=150)
    print("Gráfico guardado exitosamente como 'mi_primer_espectro.png'.")
else:
    print(f"Archivo de espectro no encontrado en: {spec_file}")
```

---

## 6. Verificación y Suite de Pruebas Automatizadas

Para comprobar que el entorno y las funciones numéricas operan con total precisión:

```bash
# Ejecutar todas las pruebas rápidas (~1.5 minutos)
python -m pytest tests/ -q -m "not slow"

# Verificar que todos los archivos compilan sin errores de sintaxis
python -m compileall musepipe tests stage08_full_spectrum_for_modeling.py
```

Si todos los tests reportan `passed`, el sistema está listo y operando en óptimas condiciones.
