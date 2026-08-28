"""El constructor de notebooks de análisis copia el código de la cadena.

Copiar el fuente en vez de importarlo es una decisión deliberada — el notebook
tiene que ser editable — pero solo se sostiene si la copia (a) es de verdad la
de `musepipe`, (b) llega completa (imports y constantes incluidos) y (c) el
notebook avisa cuando la cadena cambia. Eso es lo que se fija aquí.

La prueba de que la copia REPRODUCE la cadena la hace el propio notebook, en su
celda de comparación; el test de datos reales de abajo la ejecuta y exige que
diga «IDÉNTICO».
"""
import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_debug_notebooks", ROOT / "scripts" / "build_debug_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InlinedSourceTests(unittest.TestCase):
    """Se comprueban TODAS las etapas cubiertas, no solo la primera."""

    @classmethod
    def setUpClass(cls):
        cls.bdn = _load_builder()

    def _sources(self, stage_id):
        return self.bdn.extract_sources(stage_id)

    def test_the_copy_is_literally_the_source_in_musepipe(self):
        for stage_id in self.bdn.INLINE_SOURCES:
            for rel, name, src, _sha in self._sources(stage_id):
                with self.subTest(etapa=stage_id, funcion=name):
                    text = (ROOT / rel).read_text(encoding="utf-8")
                    self.assertIn(src, text, f"{rel}:{name} no coincide con el fuente")

    def test_the_recorded_sha_matches_the_copied_text(self):
        # Si el sha se calculara sobre otro texto (p.ej. sin recortar el salto
        # final), el chequeo de deriva daria falsa alarma nada mas generar.
        import hashlib
        for stage_id in self.bdn.INLINE_SOURCES:
            for rel, name, src, sha in self._sources(stage_id):
                with self.subTest(etapa=stage_id, funcion=name):
                    actual = hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]
                    self.assertEqual(actual, sha, f"{rel}:{name}")

    #: Lo que cada etapa TIENE que arrastrar. Por etapa y no en común: lo que
    #: se fija aquí son regresiones concretas, y exigirle a A3 el
    #: `FLAG_BAD_WINDOW` del bloque C solo mediría que la lista está mal.
    ARRASTRA = {
        # `warnings` lo usa robust_sigma_axis0 y `FLAG_BAD_WINDOW` channel_flags:
        # sin detectarlos, el notebook fallaba al ejecutarse.
        "C2": (["import warnings", "import numpy as np"], ["FLAG_BAD_WINDOW"]),
        "C3": (["import warnings", "import numpy as np"], ["FLAG_BAD_WINDOW"]),
        "C4": (["import warnings", "import numpy as np"], ["FLAG_BAD_WINDOW"]),
        "C5": (["import warnings", "import numpy as np"], ["FLAG_BAD_WINDOW"]),
        "C6": (["import warnings", "import numpy as np"], ["FLAG_BAD_WINDOW"]),
        # A3: las dos ventanas protegidas solo aparecen DENTRO del valor de
        # `PROTECTED_WINDOWS`, así que fijan el cierre transitivo de
        # `needed_constants` — sin él la copia peta con NameError.
        "A3": (["import numpy as np", "from dataclasses import dataclass"],
               ["TELLURIC_BANDS", "PROTECTED_WINDOWS",
                "HALPHA_PROTECTED", "NALGS_PROTECTED"]),
        # C1: `functools` lo trae el decorador `@lru_cache` de
        # `_psfao_image_cached` —el único caso en que el import viene de una
        # línea que no es el cuerpo de la función— y `DEFAULT_X0` es el vector
        # de arranque único que la §7 señala como causa de los 12 bins caídos.
        "C1": (["import functools", "import math", "import numpy as np",
                "import warnings", "from dataclasses import dataclass",
                "from scipy.ndimage import gaussian_filter1d",
                "from scipy.optimize import least_squares"],
               ["PSFAO_PARAM_NAMES", "_PSFAO_PARAM_NAMES", "DEFAULT_X0",
                "PSF_SHAPE_PARAMS", "MOFFAT_BETA_FLOOR"]),
        # D2: `BAD_CONTINUUM_FLAGS` está escrita en términos de dos nombres
        # IMPORTADOS, no de otras constantes. Fija que `needed_imports` mire
        # también las constantes copiadas: mirando solo los `def`, ese bloque
        # salía sin sus imports y la primera celda de código petaba.
        "D2": (["import numpy as np", "from dataclasses import dataclass",
                "from musepipe.extraction.aperture import FLAG_BAD_WINDOW",
                "from musepipe.extraction.aperture import FLAG_SKYLINE"],
               ["BAD_CONTINUUM_FLAGS"]),
        # APCORR arrastra lo de C4 (el ajuste de dos PSF) y además
        # `MAX_INTERIOR_BUMP`, que es el umbral con el que `factor_at_wavelengths`
        # decide si para la etapa. Sin él la copia peta con NameError justo en la
        # función que este notebook existe para auditar.
        "APCORR": (["import warnings", "import numpy as np"],
                   ["FLAG_BAD_WINDOW", "MAX_INTERIOR_BUMP"]),
        # `residuos_debug` no copia ninguna constante -- ninguna de sus funciones
        # se apoya en una -- y sí arrastra `gaussian_filter1d`, que es con lo que
        # `radial_hybrid_profile` suaviza el perfil: sin él la copia peta justo en
        # la función que este notebook usa de lupa.
        "RESID": (["import numpy as np", "from scipy.ndimage import gaussian_filter1d",
                   "from musepipe.stats import finite_percentile"], []),
        # PSFHALO copia lo de RESID mas la evaluacion del modelo, que arrastra
        # `math`/`functools` (psfao) y las constantes de la forma Moffat.
        "PSFHALO": (["import numpy as np", "import functools", "import math",
                     "from scipy.ndimage import gaussian_filter1d",
                     "from musepipe.stats import finite_percentile"],
                    ["PSF_SHAPE_PARAMS", "MOFFAT_BETA_FLOOR", "MIXTURE_FORM"]),
    }

    def test_the_copy_carries_the_imports_and_constants_it_uses(self):
        for stage_id in self.bdn.INLINE_SOURCES:
            sources = self._sources(stage_id)
            imports = self.bdn.needed_imports(sources)
            constants = "\n".join(self.bdn.needed_constants(sources))
            with self.subTest(etapa=stage_id):
                self.assertIn(stage_id, self.ARRASTRA,
                              "etapa nueva sin expectativa declarada en ARRASTRA")
                esperados_imports, esperadas_constantes = self.ARRASTRA[stage_id]
                for imp in esperados_imports:
                    self.assertIn(imp, imports)
                for cte in esperadas_constantes:
                    self.assertIn(cte, constants)

    def test_the_constants_are_watched_by_the_drift_check_too(self):
        """Cada constante copiada tiene su sha, y el sha es el de `musepipe`.

        El guardia solo hasheaba `FunctionDef`/`ClassDef`: las constantes
        viajaban sin vigilar, así que tocar `TELLURIC_BANDS` dejaba la copia
        atrás mientras el notebook imprimía «sin deriva».
        """
        import hashlib
        for stage_id in self.bdn.INLINE_SOURCES:
            sources = self._sources(stage_id)
            shas = self.bdn.constant_shas(sources)
            seleccion = self.bdn.selected_constants(sources)
            with self.subTest(etapa=stage_id):
                _imports, esperadas = self.ARRASTRA[stage_id]
                # Que una etapa no copie constantes es legítimo (`RESID` no usa
                # ninguna); el canario de «se rompió la detección» lo sostienen
                # las etapas que SÍ declaran constantes en ARRASTRA.
                if esperadas:
                    self.assertTrue(seleccion,
                                    "ninguna constante copiada: ¿se rompió la detección?")
                for cte in esperadas:
                    self.assertTrue(any(k.endswith(f":{cte}") for k in shas),
                                    f"{stage_id}: {cte} viaja copiada pero sin sha")
            for rel, targets, src in seleccion:
                with self.subTest(etapa=stage_id, constante=targets):
                    esperado = hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]
                    for name in targets:
                        self.assertEqual(shas[f"{rel}:{name}"], esperado)
                    self.assertIn(src, (ROOT / rel).read_text(encoding="utf-8"))

    def test_a_constant_never_precedes_what_it_is_written_in_terms_of(self):
        """Las constantes salen en orden de módulo, no de descubrimiento.

        `PROTECTED_WINDOWS = (HALPHA_PROTECTED, NALGS_PROTECTED)` se descubre
        antes que sus dos operandos; emitirla antes que ellos daría un notebook
        que compila y revienta al ejecutarse.

        Se ejecuta con los imports delante, que es como los emite el notebook:
        una constante puede estar escrita en términos de un nombre IMPORTADO
        (`BAD_CONTINUUM_FLAGS = FLAG_BAD_WINDOW | FLAG_SKYLINE` en D2) y sin
        ellos este test mediría otra cosa.
        """
        for stage_id in self.bdn.INLINE_SOURCES:
            sources = self._sources(stage_id)
            bloque = "\n".join(self.bdn.needed_imports(sources) + [""]
                               + self.bdn.needed_constants(sources))
            with self.subTest(etapa=stage_id):
                try:
                    exec(compile(bloque, f"<constantes {stage_id}>", "exec"), {})
                except NameError as exc:
                    self.fail(f"{stage_id}: la copia no define lo que usa ({exc})")

    def test_nothing_the_copy_calls_is_left_undefined(self):
        """Ninguna función copiada puede llamar a otra que no viaje con ella."""
        for stage_id, modules in self.bdn.INLINE_SOURCES.items():
            sources = self._sources(stage_id)
            inlined = {name for _r, name, _s, _h in sources}
            blob = "\n\n".join(src for _r, _n, src, _h in sources)
            called = {n.func.id for n in ast.walk(ast.parse(blob))
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            defined_upstream = set()
            for rel, _names in modules:
                tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
                defined_upstream |= {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            with self.subTest(etapa=stage_id):
                self.assertEqual(sorted((called & defined_upstream) - inlined), [])


class GeneratedNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bdn = _load_builder()
        cls.mb = cls.bdn._load_main_builder()
        cls.cells = {
            stage_id: builder(cls.mb, "ROXs12b", "ROXs12b_realigned")
            for stage_id, (_slug, builder) in cls.bdn.BUILDERS.items()
        }

    def test_every_code_cell_compiles(self):
        for stage_id, cells in self.cells.items():
            for i, cell in enumerate(cells):
                if cell["cell_type"] != "code":
                    continue
                with self.subTest(etapa=stage_id, celda=i):
                    compile("".join(cell["source"]), f"<{stage_id} celda {i}>", "exec")

    #: Nombres que IPython REASIGNA por su cuenta antes de ejecutar cada celda:
    #: `_i`/`_ii`/`_iii` son el código de las celdas anteriores, `_`/`__`/`___`
    #: las últimas salidas, `In`/`Out` el historial. Un valor guardado en uno de
    #: ellos NO sobrevive al salto de celda.
    IPYTHON_RESERVADOS = frozenset({
        "_i", "_ii", "_iii", "_", "__", "___", "_ih", "_oh", "_dh",
        "In", "Out", "_exit_code", "_sh",
    })

    def test_no_cell_reads_an_ipython_reserved_name_from_another_cell(self):
        """Un `_i` que cruza de celda funciona con `exec` y revienta en Jupyter.

        Es un fallo que NINGUNA otra prueba puede ver: el test `slow` y
        `nbclient` sobre el .ipynb ejecutan las celdas con `exec` en un espacio
        de nombres normal, donde `_i` es una variable como otra cualquiera. En
        un kernel de IPython, en cambio, `_i` se reescribe con el código de la
        celda anterior antes de cada ejecución, así que el valor se pierde al
        cambiar de celda y la siguiente falla con `IndexError`. Pasó de verdad
        en la §8 de C1 (`_i = int(np.nanargmax(_salto))` usado en la celda de
        las figuras), y solo se vio abriendo el notebook a mano.

        Asignar uno de esos nombres y consumirlo EN LA MISMA celda es correcto
        y muy común (`for _i in ...`), así que solo se persigue lo que cruza.
        """
        for stage_id, cells in self.cells.items():
            codigo = [c for c in cells if c["cell_type"] == "code"]
            asignados_antes = set()
            for i, cell in enumerate(codigo):
                arbol = ast.parse("".join(cell["source"]))
                nombres = [n for n in ast.walk(arbol) if isinstance(n, ast.Name)]
                escribe = {n.id for n in nombres if isinstance(n.ctx, ast.Store)}
                lee = {n.id for n in nombres if isinstance(n.ctx, ast.Load)}
                heredados = (lee - escribe) & self.IPYTHON_RESERVADOS & asignados_antes
                with self.subTest(etapa=stage_id, celda=i):
                    self.assertEqual(
                        heredados, set(),
                        f"{stage_id} celda {i} lee {sorted(heredados)} de una celda "
                        "anterior; IPython lo habrá reasignado. Renómbralo.",
                    )
                asignados_antes |= escribe & self.IPYTHON_RESERVADOS

    def test_it_carries_the_drift_check_and_the_comparison(self):
        productos = {"A3": ["stage00t_qc.json"],
                     # C1 no entrega un espectro: su producto es el modelo, y la
                     # comparación es contra el JSON reconstruido desde el CSV.
                     # Los dos CSV aparecen porque las dos formas viajan.
                     "C1": ["psf_model.json", "stage_e01_psfao_params.csv",
                            "stage_e01_psf_params.csv", "stage_e01_qc.json"],
                     "C2": ["spec_aperture_object.fits"],
                     "C3": ["spec_optimal_object.fits", "spec_optimal_psfsub_object.fits"],
                     "C4": ["spec_psffit_object.fits", "spec_psffit_star.fits"],
                     "C5": ["spec_sgf_object.fits"],
                     "C6": ["spec_lpm_object.fits"],
                     "D2": ["spec_calibrated_psffit_star.fits"],
                     # APCORR sigue la primaria por toda la cadena, así que se
                     # compara contra los DOS productos que la llevan: el de C4
                     # y el calibrado de D2 (que para la estrella tiene el mismo
                     # flujo, solo cambia el eje λ — y el notebook lo comprueba).
                     "APCORR": ["spec_psffit_star.fits",
                                "spec_calibrated_psffit_star.fits"],
                     # `residuos_debug` no reproduce un espectro: reconstruye el
                     # residuo `dato - modelo` desde el CSV de C1, y lo que tiene
                     # que salir idéntico es el producto que la cadena saca de ese
                     # mismo residuo — la mediana azimutal de la rama híbrida.
                     "RESID": ["psf_hybrid_residual.fits",
                               "stage_e01_psfao_params.csv", "psf_model.json"],
                     # PSFHALO no rehace una etapa: cuenta la investigacion del
                     # halo. Se ancla igual que los demas -el cociente
                     # nucleo/total del modelo contra el QC, y el perfil radial
                     # contra el producto de la rama hibrida-, porque sin ancla
                     # seria prosa y no analisis.
                     "PSFHALO": ["stage_e01_qc.json", "psf_hybrid_residual.fits",
                                 "stage_e01_psfao_params.csv",
                                 "stage02_xcorr_cube_stack.fits"]}
        for stage_id, cells in self.cells.items():
            text = "\n".join("".join(c["source"]) for c in cells)
            with self.subTest(etapa=stage_id):
                self.assertIn("chequeo_de_deriva", text)
                self.assertIn("IDÉNTICO", text)
                for producto in productos[stage_id]:
                    self.assertIn(producto, text)

    def test_the_knobs_come_from_the_resolved_stage_config(self):
        """No copiados como literales: la etapa rellena defaults que no están
        escritos en el run (C3 hereda el anillo de fondo de C2), y copiarlos a
        mano fue justo lo que hizo que C3 no reprodujera la cadena."""
        for stage_id, cells in self.cells.items():
            text = "\n".join("".join(c["source"]) for c in cells)
            with self.subTest(etapa=stage_id):
                # Con `project_root=ROOT`: musepipe resuelve rutas contra el cwd,
                # que en un notebook es su propia carpeta — sin eso, buscaba el
                # config bajo `notebooks/<obj>/debug/runs/...` y reventaba.
                # `_from_run(...)` y no el nombre completo: C5/C6 lo importan con
                # alias porque comparten constructor.
                self.assertIn("_from_run(RUN_ID, project_root=ROOT)", text)

    def test_it_writes_a_valid_notebook_under_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "C2_aperture_debug.ipynb"
            path.write_text(
                json.dumps(self.mb.notebook(self.cells["C2"]), indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["nbformat"], 4)
        self.assertTrue(payload["cells"])
        self.assertEqual(payload["metadata"]["kernelspec"]["name"], "python3")


@pytest.mark.slow
class ReproducesTheChainTests(unittest.TestCase):
    """Ejecuta cada notebook de análisis y exige que reproduzca su etapa.

    Es el único test que prueba de verdad que la copia sigue siendo la cadena:
    lo demás verifica que el texto coincide, esto verifica que los NÚMEROS
    coinciden. Cuesta ~1.5 min por notebook, de ahí el marcador `slow`.

    Se salta si el run no está en disco (`runs/` no se versiona), que es el caso
    de un clon limpio.

    **La §10 de C1 tiene caché, y la primera vez la llena.** Ajusta las dos
    formas de PSF en cada cubo por exposición y guarda el resultado en
    `runs/<RUN>/tables/c1_perexp_psf_params.json`; con las perillas por defecto
    eso son ~320 ajustes a ~4.5 s, o sea ~24 min la PRIMERA ejecución en una
    máquina que tenga los cubos delante. Las siguientes leen la caché en menos
    de un segundo y el notebook vuelve a costar lo de siempre. En una máquina
    sin esos cubos la sección se salta sola y no cuesta nada. Si este test
    parece colgado, es eso: mira si la caché está creciendo.
    """

    #: notebook -> productos de la etapa que tienen que existir para compararlo
    CASOS = {
        "A3_telluric_debug": ["stage00t_qc.json"],
        # C1 necesita además el cubo sobre el que ajustó (§6 reajusta bins de
        # verdad y §9.b mide la apertura sobre él) y el producto de C4, que es
        # el coeficiente contra el que se contrasta el reparto núcleo/halo. El
        # CSV de psfao NO se exige: solo existe si psfao ganó, y el notebook
        # audita la forma que la etapa eligió (ROXs 42B b sale moffat).
        "C1_chromatic_psf_debug": ["psf_model.json", "stage_e01_psf_params.csv",
                                   "stage_e01_qc.json",
                                   "stage02_xcorr_cube_stack.fits", "spec_psffit_star.fits"],
        "C2_aperture_debug": ["spec_aperture_object.fits"],
        "C3_optimal_debug": ["spec_optimal_object.fits", "spec_optimal_psfsub_object.fits"],
        "C4_psffit_debug": ["spec_psffit_object.fits", "spec_psffit_star.fits"],
        "C5_sgf_debug": ["spec_sgf_object.fits"],
        "C6_lpm_debug": ["spec_lpm_object.fits"],
        "D2_primary_star_debug": ["spec_psffit_star.fits", "spec_calibrated_psffit_star.fits"],
        # `apcorr_debug` re-ajusta el psffit sobre el cubo de B2, así que además
        # de los productos necesita el cubo — como C1.
        "apcorr_debug": ["spec_psffit_star.fits", "spec_calibrated_psffit_star.fits",
                         "stage02_xcorr_cube_stack.fits", "psf_model.json",
                         "stage01c_qc.json"],
        # `residuos_debug` reconstruye los modelos por bin desde el CSV, asi que
        # necesita el cubo (las imagenes por bin) ademas del producto que compara.
        # El CSV de psfao NO se exige: solo existe si psfao gano, y el notebook
        # dice que no aplica cuando la forma elegida es Moffat.
        "residuos_debug": ["psf_hybrid_residual.fits", "psf_model.json",
                           "stage_e01_qc.json", "stage01c_qc.json",
                           "stage02_xcorr_cube_stack.fits"],
        "psf_halo_cromatico_debug": ["psf_hybrid_residual.fits", "psf_model.json",
                                     "stage_e01_qc.json", "stage01c_qc.json",
                                     "stage02_xcorr_cube_stack.fits"],
    }
    #: (objeto, run) de cada cadena. Estaba fijado a ROXs 12 b, asi que los cinco
    #: notebooks de ROXs 42B b **no los ejecutaba nadie**: se generaban y nadie
    #: comprobaba que reprodujeran su etapa. Cada objeto se salta solo si le
    #: faltan los productos, asi que un clon limpio sigue pasando.
    OBJETOS = (("ROXs12b", "ROXs12b_realigned"), ("ROXs42Bb", "ROXs42Bb_realigned"))

    @staticmethod
    def _entrada_fuera_del_run(slug, stage_dir):
        """Ruta que falta, de las que NO viven bajo `runs/<RUN>/stages`.

        Los notebooks del bloque C se comparan contra un producto del propio
        run; A3 mide sobre los cubos de reducción, que están fuera del árbol
        del run (y en otro disco). Sin esta comprobación el notebook se
        ejecutaría y fallaría por falta de datos en vez de saltarse.
        """
        if slug != "A3_telluric_debug":
            return None
        qc = stage_dir / "stage00t_qc.json"
        if not qc.exists():
            return str(qc)
        declarado = (json.loads(qc.read_text(encoding="utf-8"))
                     .get("input", {}).get("cube", ""))
        cubo = Path(declarado) if declarado else None
        if cubo is not None and not cubo.is_absolute():
            cubo = ROOT / cubo
        return None if (cubo is not None and cubo.exists()) else str(declarado)

    #: notebook -> (productos de la etapa, entrada de la que se derivan). El
    #: notebook solo puede salir IDÉNTICO si el producto se escribió DESPUÉS de
    #: su entrada; si la etapa de arriba se re-ejecutó y la de abajo no, lo que
    #: hay en disco calibra un espectro que ya no existe y la diferencia no
    #: mide la copia, mide que el run está a medias.
    #:
    #: El primer elemento puede ser un nombre o varios: manda el MÁS VIEJO, que
    #: es el que delata que la etapa no se re-ejecutó entera.
    DERIVADOS = {
        "D2_primary_star_debug": ("spec_calibrated_psffit_star.fits", "spec_psffit_star.fits"),
        # C1: la §6 reajusta bins del cubo y los compara contra el CSV. Si el
        # cubo cambió y C1 no se ha vuelto a correr, esos dos números salen de
        # datos distintos y el DIFIERE no mide la copia.
        "C1_chromatic_psf_debug": ("psf_model.json", "stage02_xcorr_cube_stack.fits"),
        # C2-C6 consumen el modelo de PSF de C1. Re-correr C1 sin re-correr el
        # bloque C deja a estos notebooks recalculando con el modelo NUEVO
        # contra productos hechos con el viejo: DIFIERE garantizado, y no mide
        # la copia. Pasó de verdad el 2026-08-08 (`docs/2026-08-09_handoff.md`
        # §5.1): `psf_model.json` de las 13:02 y los espectros de la noche
        # anterior, y los cinco subtests se pusieron en rojo sin que hubiera
        # cambiado una línea de la cadena.
        #
        # Y ademas de `psf_model.json`, el MODULO que calcula la correccion de
        # apertura: los seis metodos multiplican por `factor_at_wavelengths`, asi
        # que cambiarlo deja igual de obsoletos a sus productos. Paso el
        # 2026-08-11 al pasar de parabola a PCHIP: los cinco notebooks de
        # ROXs 42B b se pusieron en rojo sin que su copia tuviera nada malo.
        "C2_aperture_debug": ("spec_aperture_object.fits",
                              ("psf_model.json", "musepipe/growth_curve.py")),
        "C3_optimal_debug": (("spec_optimal_object.fits",
                              "spec_optimal_psfsub_object.fits"),
                             ("psf_model.json", "musepipe/growth_curve.py")),
        "C4_psffit_debug": (("spec_psffit_object.fits", "spec_psffit_star.fits"),
                            ("psf_model.json", "musepipe/growth_curve.py")),
        "C5_sgf_debug": ("spec_sgf_object.fits",
                         ("psf_model.json", "musepipe/growth_curve.py")),
        "C6_lpm_debug": ("spec_lpm_object.fits",
                         ("psf_model.json", "musepipe/growth_curve.py")),
        # `apcorr_debug` recalcula la apcorr con el CODIGO DE HOY. Si el producto
        # se escribio antes de que ese codigo cambiara, la diferencia mide que el
        # run esta por detras de la cadena, no la copia. Por eso su entrada no es
        # un producto del run sino un fichero del repo: paso el 2026-08-11 al
        # cambiar el interpolador de las bandas (parabola -> PCHIP), que dejo los
        # productos de ROXs 42B b describiendo una apcorr que ya no se calcula
        # asi.
        # A la entrada de codigo se le suma el `psf_model.json` que ya declaran
        # C2-C6: apcorr_debug tambien EVALUA el modelo de C1 (`fit_psffit_cube`
        # y `aperture_correction_from_psf`), asi que un modelo mas nuevo que el
        # producto deja la comparacion midiendo el desfase del run. Paso el
        # 2026-08-17 al re-ejecutar C1 con `psf_scope=per_observation`: sus seis
        # hermanos se saltaron y este fallo, por no tener declarada la entrada.
        "apcorr_debug": (("spec_psffit_star.fits", "spec_calibrated_psffit_star.fits"),
                         ("psf_model.json", "musepipe/growth_curve.py")),
    }

    @classmethod
    def _producto_desfasado(cls, slug, stage_dir):
        """`None`, o el motivo por el que el producto en disco no es comparable."""
        par = cls.DERIVADOS.get(slug)
        if par is None:
            return None
        nombres, entrada_n = par
        if isinstance(nombres, str):
            nombres = (nombres,)
        # Una entrada con `/` es una ruta del REPO (p. ej. el modulo que calcula
        # la correccion), no un producto del run: la frescura tambien se pierde
        # cuando cambia el codigo, no solo cuando cambia un fichero de entrada.
        if isinstance(entrada_n, str):
            entrada_n = (entrada_n,)
        entradas = [(ROOT / n) if "/" in n else (stage_dir / n) for n in entrada_n]
        entradas = [e for e in entradas if e.exists()]
        salidas = [stage_dir / n for n in nombres]
        salidas = [s for s in salidas if s.exists()]
        if not salidas or not entradas:
            return None
        # Manda la entrada MAS NUEVA contra la salida MAS VIEJA: basta con que
        # una entrada se haya movido despues para que el producto ya no sirva.
        salida = min(salidas, key=lambda s: s.stat().st_mtime)
        entrada = max(entradas, key=lambda e: e.stat().st_mtime)
        if salida.stat().st_mtime >= entrada.stat().st_mtime:
            return None
        return (f"{salida.name} es más viejo que {entrada.name}: la etapa no se ha "
                "re-ejecutado desde su entrada")

    def test_every_debug_notebook_reports_identical(self):
        import matplotlib
        matplotlib.use("Agg")

        for objeto, run in self.OBJETOS:
            stage_dir = ROOT / "runs" / run / "stages"
            for slug, productos in self.CASOS.items():
                path = ROOT / "notebooks" / objeto / "debug" / f"{slug}.ipynb"
                faltan = [n for n in productos if not (stage_dir / n).exists()]
                with self.subTest(objeto=objeto, notebook=slug):
                    if faltan:
                        self.skipTest(f"{run} sin productos: falta {faltan[0]}")
                    fuera = self._entrada_fuera_del_run(slug, stage_dir)
                    if fuera:
                        self.skipTest(f"{slug}: entrada fuera del run sin disponer: {fuera}")
                    desfasado = self._producto_desfasado(slug, stage_dir)
                    if desfasado:
                        self.skipTest(f"{slug}: {desfasado}")
                    if not path.exists():
                        self.skipTest(f"{slug} no generado")
                    self._reproduce_la_cadena(path, slug)

    def _reproduce_la_cadena(self, path, slug):
        """Ejecuta el notebook y exige que diga «sin deriva» e «IDÉNTICO»."""
        import contextlib
        import io
        import os

        payload = json.loads(path.read_text(encoding="utf-8"))
        namespace = {"__name__": "__main__"}
        out = io.StringIO()
        # Se ejecuta DESDE LA CARPETA DEL NOTEBOOK, que es el cwd real en
        # Jupyter. Correrlo desde la raíz del repo escondía un fallo:
        # musepipe resuelve rutas contra el cwd y `stage_xNN_config_from_run`
        # buscaba el config bajo `notebooks/<obj>/debug/runs/...`.
        cwd = os.getcwd()
        try:
            os.chdir(path.parent)
            with contextlib.redirect_stdout(out):
                for i, cell in enumerate(payload["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    exec(compile("".join(cell["source"]), f"<{slug} celda {i}>", "exec"),
                         namespace)
        finally:
            os.chdir(cwd)
        text = out.getvalue()
        self.assertIn("sin deriva", text, "la copia no coincide con musepipe")
        self.assertIn("IDÉNTICO: la copia reproduce la cadena.", text,
                      f"{slug} no reprodujo la cadena:\n{text[-1200:]}")


if __name__ == "__main__":
    unittest.main()
