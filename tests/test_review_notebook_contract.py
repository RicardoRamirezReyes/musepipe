"""Contract test for the review-notebook builder (scripts/build_review_notebooks.py).

Validates that every `exec` entry in STAGES points to a real, callable entry
point with the calling convention the generated notebooks use. This is the
test that would have caught the C_04b regression where `run_stage04b('$RUN')`
passed the run id as the `config` positional argument.
"""
import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_review_notebooks", ROOT / "scripts" / "build_review_notebooks.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


def test_stage_ids_and_slugs_unique(builder):
    ids = [s["id"] for s in builder.STAGES]
    slugs = [s["slug"] for s in builder.STAGES]
    assert len(set(ids)) == len(ids)
    assert len(set(slugs)) == len(slugs)


def test_exec_entries_match_real_entry_points(builder):
    for s in builder.STAGES:
        ex = s["exec"]
        kind = ex["kind"]
        if kind == "audit":
            continue
        if kind == "script":
            path = ROOT / "scripts" / ex["target"]
            assert path.exists(), f"{s['id']}: falta scripts/{ex['target']}"
            assert '"$@"' in path.read_text(), (
                f"{s['id']}: scripts/{ex['target']} no reenvía argumentos "
                "(el notebook le pasa --run-id)"
            )
        elif kind == "pyscript":
            path = ROOT / "scripts" / ex["target"]
            assert path.exists(), f"{s['id']}: falta scripts/{ex['target']}"
            assert "--run-id" in path.read_text(), (
                f"{s['id']}: scripts/{ex['target']} no acepta --run-id"
            )
        elif kind == "module_main":
            mod = importlib.import_module(ex["target"])
            main = getattr(mod, "main", None)
            assert callable(main), f"{s['id']}: {ex['target']} no define main()"
        elif kind == "module_run":
            mod = importlib.import_module(ex["target"])
            fn = getattr(mod, ex["fn"], None)
            assert callable(fn), f"{s['id']}: {ex['target']}.{ex['fn']} no existe"
            params = list(inspect.signature(fn).parameters.values())
            assert params and params[0].name == "run_id", (
                f"{s['id']}: {ex['target']}.{ex['fn']} tiene primer parámetro "
                f"{params[0].name if params else '(ninguno)'!r}, no 'run_id' — "
                "el notebook lo llama posicionalmente con el run id"
            )
        elif kind == "cli":
            # Literal command(s) with fixed paths (no --run-id). Every
            # `python -m <module>` invoked must import and expose main().
            cmd = ex["cmd"]
            assert cmd.strip(), f"{s['id']}: exec.cmd vacío"
            tokens = cmd.replace("\\\n", " ").split()
            modules = [tokens[i + 1] for i, tok in enumerate(tokens)
                       if tok == "-m" and i + 1 < len(tokens)]
            assert modules, f"{s['id']}: exec.cmd no invoca 'python -m <module>'"
            for target in modules:
                mod = importlib.import_module(target)
                assert callable(getattr(mod, "main", None)), (
                    f"{s['id']}: {target} no define main()"
                )
        elif kind == "launch":
            # Etapas de reducción: el comando lo arma `_nbcommon.launch_command`
            # desde las plantillas del registro. Se comprueba que cada plantilla
            # apunta a un ejecutable que existe de verdad.
            from musepipe import stage_registry as reg

            stage = reg.by_id(s["id"])
            assert stage is not None and stage.launch, (
                f"{s['id']}: exec.kind=launch pero el registro no trae plantillas"
            )
            for profile, commands in stage.launch.items():
                assert commands, f"{s['id']}/{profile}: secuencia de comandos vacía"
                for n, template in enumerate(commands, 1):
                    donde = f"{s['id']}/{profile}[{n}]"
                    tokens = template.split()
                    if tokens[0] == "bash":
                        path = ROOT / tokens[1]
                        assert path.exists(), f"{donde}: falta {tokens[1]}"
                        assert '"$@"' in path.read_text(), (
                            f"{donde}: {tokens[1]} no reenvía argumentos"
                        )
                    elif "-m" in tokens:
                        target = tokens[tokens.index("-m") + 1]
                        mod = importlib.import_module(target)
                        assert callable(getattr(mod, "main", None)), (
                            f"{donde}: {target} no define main()"
                        )
                    else:
                        path = ROOT / tokens[1]
                        assert path.exists(), f"{donde}: falta {tokens[1]}"
                    # Lo que este invariante siempre quiso cazar es un comando que
                    # corra contra el run EQUIVOCADO. Exigir `--run-id` literal ya
                    # no vale: los subcomandos de A4 que escriben M1/M2, M4/M5 y lo
                    # derivado no aceptan esa bandera, y fijan el run por la ruta
                    # del QC. Vale cualquiera de las dos formas de fijarlo; ninguna
                    # de las dos es un fallo.
                    fija_el_run = ("--run-id {run_id}" in template
                                   or "{stage_dir}" in template
                                   or "{run_dir}" in template)
                    assert fija_el_run, (
                        f"{donde}: no fija el run (ni --run-id {{run_id}} ni una ruta "
                        f"bajo {{stage_dir}}/{{run_dir}}): correría contra el run activo"
                    )
        else:
            raise AssertionError(f"{s['id']}: exec.kind desconocido {kind!r}")


def test_generated_code_cells_are_valid_python(builder):
    for s in builder.STAGES:
        nb = builder.notebook(builder.build_cells(s))
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            try:
                ast.parse(src)
            except SyntaxError as exc:  # pragma: no cover - mensaje de diagnóstico
                raise AssertionError(
                    f"{s['id']} celda {i}: código generado inválido: {exc}\n{src}"
                ) from exc
