"""Bitácora de unidades de trabajo reanudables, con sus tiempos.

Ninguna etapa de la cadena hacía checkpoint interno: la unidad de reanudación era
la etapa entera, y lo que se cortaba se perdía (el `requires_checkpoint` de E4 es
un **portero** —se niega a arrancar si la estimación pasa de 4 h— no un guardado).
Con A3 ajustando hasta 30 veces por run y aplicando después sobre cubos de miles
de canales, eso no vale.

Esto baja la granularidad un nivel **con la disciplina que `scripts/rerun_chain.py`
ya tiene probada**, y por eso se parece tanto: se acumula en vez de pisarse, se
guarda `rc` y `segundos` por unidad, y una unidad no cuenta como hecha solo porque
la bitácora lo diga. Ahí `rerun_chain` exige además que el QC siga en disco y que
su mtime no haya retrocedido (restaurar un snapshot devuelve la etapa a la cola);
aquí se exige lo mismo **más el hash de los parámetros**, que es lo que distingue
«ya está hecho» de «está hecho con otra receta». Sin el hash, cambiar un knob y
reanudar te devuelve el resultado viejo sin avisar, que es exactamente el fallo
silencioso que la cadena lleva un mes cazando.

El fichero `PARAR` es la otra mitad: pararse limpio entre unidades deja la
bitácora consistente, y matar el proceso a mitad de una unidad solo pierde esa.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping


#: Nombre del fichero-señal, igual que en `rerun_chain.py`: quien ya sabe parar
#: una re-ejecución no tiene que aprender un segundo mecanismo para parar A3.
STOP_FILE = "PARAR"


def params_hash(payload: Mapping[str, Any]) -> str:
    """Huella de los parámetros con los que se hizo una unidad.

    `sort_keys` y `default=str` a propósito: el hash tiene que ser estable entre
    ejecuciones (un dict que cambia de orden no es una receta distinta) y no puede
    reventar porque en el payload viaje un `Path` o un `numpy.float64`.
    """

    texto = json.dumps(dict(payload), sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]


@dataclass
class Unidad:
    """Una unidad de trabajo terminada, tal como se guarda."""

    clave: str
    rc: int
    segundos: float
    producto: str = ""
    params: str = ""
    mtime: float = 0.0
    timestamp_utc: str = ""
    extra: dict[str, Any] = None  # type: ignore[assignment]

    def as_dict(self) -> dict[str, Any]:
        fila = {
            "clave": self.clave,
            "rc": int(self.rc),
            "segundos": round(float(self.segundos), 2),
            "producto": self.producto,
            "params": self.params,
            "mtime": self.mtime,
            "timestamp_utc": self.timestamp_utc,
        }
        fila.update(self.extra or {})
        return fila


class Ledger:
    """La bitácora de una fase: qué está hecho, con qué receta y cuánto costó."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows: list[dict[str, Any]] = []
        if self.path.exists():
            try:
                cargado = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # Una bitácora truncada (el corte llegó mientras se escribía) no
                # puede tumbar la reanudación: se descarta y se rehace el trabajo,
                # que es lo conservador. Perder tiempo es aceptable; saltarse una
                # unidad por leer basura, no.
                cargado = []
            if isinstance(cargado, list):
                self.rows = [fila for fila in cargado if isinstance(fila, dict)]

    # -- lectura ---------------------------------------------------------

    def fila(self, clave: str) -> dict[str, Any] | None:
        """La última fila de esa clave: se acumula, así que la que manda es la de atrás."""

        for fila in reversed(self.rows):
            if fila.get("clave") == clave:
                return fila
        return None

    def hecha(self, clave: str, *, params: str | None = None) -> bool:
        """¿Se puede saltar esta unidad? Las cuatro condiciones, todas.

        No basta con que la bitácora diga `rc=0`: el producto tiene que seguir en
        disco, no haber retrocedido en el tiempo, y haberse hecho con **estos**
        parámetros.
        """

        fila = self.fila(clave)
        if fila is None or int(fila.get("rc", 1)) != 0:
            return False
        if params is not None and str(fila.get("params", "")) != params:
            return False
        producto = str(fila.get("producto") or "")
        if producto:
            ruta = Path(producto)
            if not ruta.exists():
                return False
            marca = fila.get("mtime")
            if marca is not None and ruta.stat().st_mtime + 1e-6 < float(marca):
                return False
        return True

    def segundos_totales(self) -> float:
        return float(sum(float(fila.get("segundos", 0.0)) for fila in self.rows))

    # -- escritura -------------------------------------------------------

    def anota(self, clave: str, *, rc: int, segundos: float, producto: str | Path = "",
              params: str = "", **extra: Any) -> dict[str, Any]:
        """Guarda una unidad y **vuelca el fichero en el acto**.

        El volcado es por unidad y no al final a propósito: una bitácora que solo
        se escribe al terminar no sirve para reanudar nada.
        """

        ruta = Path(producto) if producto else None
        unidad = Unidad(
            clave=clave,
            rc=int(rc),
            segundos=float(segundos),
            producto=str(producto) if producto else "",
            params=params,
            mtime=ruta.stat().st_mtime if ruta is not None and ruta.exists() else 0.0,
            timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            extra=dict(extra),
        )
        fila = unidad.as_dict()
        self.rows.append(fila)
        self.volcar()
        return fila

    def volcar(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.rows, indent=1, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        return self.path

    # -- parada limpia ---------------------------------------------------

    def parar(self) -> bool:
        """¿Hay un `PARAR` junto a la bitácora? Se mira **entre** unidades."""

        return (self.path.parent / STOP_FILE).exists()


class Cronometro:
    """`with Cronometro() as c: ...` y después `c.segundos`. Nada más."""

    def __init__(self) -> None:
        self.segundos = 0.0
        self._t0 = 0.0

    def __enter__(self) -> "Cronometro":
        self._t0 = time.time()
        return self

    def __exit__(self, *_exc) -> bool:
        self.segundos = time.time() - self._t0
        return False


def bloque_timing(rows: Iterable[Mapping[str, Any]], *, unidades_totales: int | None = None,
                  **tasas: float) -> dict[str, Any]:
    """El bloque `timing` del QC, con la forma del `runtime_budget` de E4.

    Se parece al de E4 aposta: aquel ya distingue estimado de medido y publica
    `estimate_over_measured`, y tener dos formatos para lo mismo es cómo se acaba
    sin poder comparar dos runs.
    """

    # Se deduplica por clave, y gana la ultima: la bitacora **se acumula** entre
    # relanzamientos (a proposito), asi que contar filas cuenta reintentos y no
    # unidades. Es la misma regla que usa `hecha()` — la fila que manda es la de
    # atras—, y sin ella el `timing` publica 22 unidades hechas de 8 totales.
    por_clave: dict[str, dict[str, Any]] = {}
    for fila in rows:
        por_clave[str(fila.get("clave"))] = dict(fila)
    filas = list(por_clave.values())
    segundos = [float(fila.get("segundos", 0.0)) for fila in filas]
    hechas = len(filas)
    total = float(sum(segundos))
    bloque: dict[str, Any] = {
        "n_unidades_hechas": hechas,
        "n_unidades_totales": int(unidades_totales) if unidades_totales is not None else hechas,
        "segundos_totales": round(total, 1),
        "horas_totales": round(total / 3600.0, 3),
        "segundos_por_unidad": round(total / hechas, 2) if hechas else None,
        "segundos_min": round(min(segundos), 2) if segundos else None,
        "segundos_max": round(max(segundos), 2) if segundos else None,
        "por_unidad": {str(fila.get("clave")): round(float(fila.get("segundos", 0.0)), 2)
                       for fila in filas},
    }
    if unidades_totales and hechas and hechas < int(unidades_totales):
        restantes = int(unidades_totales) - hechas
        bloque["segundos_restantes_estimados"] = round(restantes * total / hechas, 1)
    for nombre, valor in tasas.items():
        bloque[nombre] = None if valor is None else round(float(valor), 4)
    return bloque
