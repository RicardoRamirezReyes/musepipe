# 01. Guía Conceptual del Pipeline (Sin Jerga)

> [!IMPORTANT]
> **AVISO PARA AGENTES DE IA / AI AGENT NOTICE**:
> Este documento corresponde a material explicativo y formativo redactado para el **usuario humano**. **NO constituye instrucciones ni órdenes operacionales para agentes de IA**.
> Las directivas y reglas para agentes están gobernadas exclusivamente por [`AGENTS.md`](../../AGENTS.md).

---

## 1. Introducción al Problema Científico

Imagina que intentas fotografiar una luciérnaga posada a diez centímetros de un reflector de estadio encendido. A simple vista, el resplandor del reflector inunda toda la cámara, ocultando por completo la tenue luz del insecto.

En astrofísica observacional ocurre exactamente lo mismo:
- **La Estrella Primaria**: Es el "reflector". Una estrella joven masiva o similar al Sol, miles o millones de veces más brillante.
- **El Compañero**: Es la "luciérnaga". Un planeta gigante joven en formación o una enana marrón orbitando cerca de la estrella.
- **El Fenómeno de Acreción**: El planeta está "comiendo" gas y polvo de su entorno. Cuando ese gas cae a velocidades extremas hacia el objeto, se calienta y emite un destello de luz muy particular en un color específico (la línea de emisión de Hidrógeno Alfa, $H\alpha$).

El objetivo de este proyecto es tomar imágenes astronómicas tridimensionales, eliminar el resplandor de la estrella primaria con modelos matemáticos de alta precisión, extraer la luz pura del compañero y medir si está emitiendo esa señal de acreción.

---

## 2. ¿Qué es un Cubo 3D de Datos MUSE?

Una cámara digital normal toma imágenes en dos dimensiones espaciales: ancho ($x$) y alto ($y$), con 3 canales de color (Rojo, Verde, Azul).

El instrumento **MUSE** (Multi Unit Spectroscopic Explorer) es un **espectrógrafo de campo integral**. En lugar de 3 colores, divide la luz en **3.681 colores extremadamente precisos** (longitudes de onda desde $4750\,\text{Å}$ hasta $9350\,\text{Å}$, con pasos de $1.25\,\text{Å}$).

En términos de programación en Python, un cubo de datos MUSE no es más que un arreglo NumPy tridimensional:

```python
# data es una matriz NumPy 3D leída desde un archivo FITS
# data.shape -> (3681, 100, 100)
#                ^^^^  ^^^  ^^^
#                |     |    |
#                |     |    +--> Coordenada espacial X en el cielo (píxeles)
#                |     +-------> Coordenada espacial Y en el cielo (píxeles)
#                +-------------> Eje de Longitud de Onda (Lambda / Z)
```

```
           Z (Longitud de Onda: 4750 Å -> 9350 Å)
          /
         /   +-------------------+
        /   /                   /|
       /   /                   / |
      /   /   Imagen a 6563 Å /  |
     +---+-------------------+   |
     |   |                   |   |
   Y |   |   (Corte 2D)      |   +
     |   |                   |  /
     |   |                   | /
     +---+-------------------+/
               X
```

- **Si haces un corte en `Z`** (`data[1450, :, :]`): Obtienes una foto 2D del cielo tomada exactamente en una longitud de onda específica (por ejemplo, a $6562.8\,\text{Å}$).
- **Si seleccionas un píxel espacial `(Y, X)`** (`data[:, 50, 50]`): Obtienes un vector 1D de 3.681 elementos que representa el **espectro completo de luz** emitido por ese punto del cielo. A cada píxel espacial se le denomina **spaxel** (*spatial pixel*).

---

## 3. La PSF (Point Spread Function) y la Turbulencia Atmosférica

Si observáramos una estrella puntual desde el espacio con un telescopio perfecto, veríamos un punto casi nítido. Sin embargo, la luz estelar atraviesa la atmósfera de la Tierra, donde corrientes de aire frío y caliente actúan como lentes deformantes que hacen que la estrella "parpadee" y se desenfoque.

A la mancha de luz resultante se le llama **PSF** (*Point Spread Function* o Función de Dispersión de Punto):
1. **Núcleo central**: Donde se concentra la mayor parte de la luz.
2. **Alas o Halo**: Un resplandor difuso que se extiende cientos de píxeles alrededor de la estrella.

El instrumento MUSE cuenta con un sistema de **Óptica Adaptativa** (láseres que miden la turbulencia en tiempo real y deforman espejos 1.000 veces por segundo para corregirla). Aun así, queda un halo residual que ahoga la luz del compañero.

### ¿Cómo modela el pipeline la PSF?
El módulo `musepipe/psf.py` ajusta modelos matemáticos a la forma de la estrella:
- **Perfil de Moffat**: Una campana matemática flexible con un núcleo y alas extendidas.
- **Modelo Psfao**: Un modelo físico avanzado que incluye la corrección del sistema de óptica adaptativa.

Dado que la atmósfera distorsiona los colores azules de forma distinta a los rojos, el tamaño de la PSF cambia con la longitud de onda. Por eso el pipeline calcula una **PSF cromática** (etapa `C1`).

---

## 4. Los Seis Métodos para "Desenterrar" al Compañero

Para separar la luz del compañero del halo de la estrella, el pipeline implementa **seis métodos numéricos distintos** (etapas C2 a C6):

```
+------------------+-------------------------------------------------------------+
| Método           | ¿Cómo funciona intuitivamente?                              |
+------------------+-------------------------------------------------------------+
| 1. aperture      | Suma los píxeles dentro de un círculo centrado en el        |
|                  | compañero y resta el fondo promedio medido en un anillo.   |
| 2. optimal_ls    | Pesa cada píxel según la forma esperada del compañero y     |
|                  | usa un ajuste de superficie plana local como fondo.        |
| 3. optimal_psfsub| Igual que el anterior, pero usando el modelo de la PSF de   |
|                  | la estrella primaria como fondo.                            |
| 4. psffit        | Ajusta simultáneamente dos PSF (una para la estrella y otra |
|                  | para el compañero) mediante regresión lineal por mínimos    |
|                  | cuadrados canal a canal. (Método canónico de referencia).   |
| 5. sgf           | Extrae el perfil radial de la estrella, lo filtra con       |
|                  | polinomios de Savitzky-Golay y mide el residuo.            |
| 6. lpm           | Modela el fondo estelar mediante polinomios de Legendre,    |
|                  | enmascarando las líneas de emisión para no falsearlas.      |
+------------------+-------------------------------------------------------------+
```

El pipeline compara los 6 resultados entre sí (etapa `D1`). Si todos los métodos convergen en la misma señal, tenemos alta certeza estadística de que el resultado es real y no un artefacto de un algoritmo particular.

---

## 5. Acreción y la Línea $H\alpha$

Cuando los electrones de los átomos de hidrógeno caen del tercer al segundo nivel de energía, emiten un fotón con una longitud de onda exacta de:

$$\lambda_{H\alpha} = 6562.8\,\text{Å}\quad (656.28\,\text{nm})$$

- Si el compañero es un objeto frío inerte (como un planeta maduro), su espectro mostrará un continuo suave y oscuro en esa zona.
- Si el compañero está acretando material activamente, veremos un **pico estrecho e intenso** que sobresale notablemente en $6562.8\,\text{Å}$.

```
  Flujo de Luz
       ^
       |                      | (Pico de emisión H-alpha por acreción)
       |                     / \
       |                    /   \
       |                   /     \
       |  ----------------+       +---------------- (Continuo estelar débil)
       +---------------------------------------------------->
                                6563 Å                   Longitud de Onda
```

Las etapas `E1`, `G2` y `G3` se encargan de detectar este pico, medir su área (flujo integrado), calcular su ancho equivalente ($EW$) y convertir esa energía en una tasa física de acumulación de masa ($\dot{M}$, masa por año).

---

## 6. El Modelo de Ruido y las Aperturas de Control

En software estándar de astronomía, los cubos incluyen una matriz llamada `STAT` con la varianza estimada de cada píxel. 

> [!CAUTION]
> **Regla de oro del proyecto**: Nunca se debe usar `STAT` directamente como el error $\sigma$, ni asumir que los píxeles vecinos son estadísticamente independientes.

¿Por qué?
1. **Remuestreo espacial**: Durante la reconstrucción del cubo 3D a partir de las imágenes crudas del detector, los píxeles se interpolan. Esto hace que píxeles adyacentes compartan información (*ruido correlacionado*), subestimando el ruido real en un factor de aproximadamente **$4\times$ a $6.5\times$**.
2. **Variación radial del halo**: El ruido cerca de la estrella brillante es mucho mayor que en el fondo del cielo lejano.

### La solución: Aperturas de Control
Para saber si una señal en el compañero es estadísticamente significativa (por ejemplo, a nivel $3\sigma$ o $5\sigma$):
1. Se mide la distancia $R$ desde la estrella primaria hasta el compañero.
2. Se colocan múltiples círculos de prueba (**aperturas de control**) a lo largo del mismo círculo de radio $R$, donde el brillo del halo estelar es idéntico pero no hay ningún objeto.
3. Se aplica exactamente el mismo método de extracción a todas las aperturas de control.
4. La desviación estándar empírica entre las aperturas de control es la verdadera barra de error $\sigma$.

```
                        (Estrella Primaria)
                                *
                             /  |  \
                            /   |   \  Radio R idéntico
                   Control /    |    \ Control
                          O     |     O
                                |
                                v
                           (Compañero)
                                ●
```

Si la señal del compañero supera con creces la dispersión de las aperturas de control, sabemos con total rigor que el descubrimiento es genuino.
