"""Static check: no function may reference a name nothing defines.

v2.17.0 added a `netload_hourly` reference to
`SpotPriceCoordinator._apply_pipeline_pre_dk` without adding the
parameter. Python resolves that at call time, not import time, so the
module imported cleanly and the method raised `NameError` on every
coordinator cycle. The coordinator's own caller catches broad exceptions
and logs a warning, so the whole prediction pipeline was dead from
v2.17.0 through v2.17.2 — no forecasts overwritten, no calibrator state
saved, and the DtACI bundles frozen — while the integration otherwise
looked healthy.

Nothing caught it because the coordinator needs a running Home Assistant
to import, so no unit test exercises that method. This check needs
neither an import nor HA: `symtable` reports, per scope, which names are
read without being bound locally. Any such name must resolve to a
module-level definition or a builtin, or it is a latent NameError.
"""
from __future__ import annotations

import builtins
import symtable
from pathlib import Path

import pytest

_PKG = Path(__file__).parent.parent / "custom_components" / "spot_price_predictor"
_MODULES = sorted(_PKG.glob("*.py"))

_IMPLICIT = {
    # Module dunders the interpreter always binds.
    "__file__", "__name__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__debug__",
    # Closure cell the compiler injects into methods using zero-arg super().
    "__class__",
}


def _module_level_names(table: symtable.SymbolTable) -> set[str]:
    """Every name bound at module scope — assignments, imports, defs,
    classes — plus builtins."""
    names = {s.get_name() for s in table.get_symbols()
             if s.is_assigned() or s.is_imported() or s.is_namespace()}
    return names | set(dir(builtins)) | _IMPLICIT


def _walk(table: symtable.SymbolTable, defined: set[str],
          path: str = "") -> list[str]:
    """Collect `scope: name` for every free/global read with no binding.

    A nested scope inherits the enclosing scope's local bindings, so
    `defined` accumulates on the way down.
    """
    problems: list[str] = []
    for child in table.get_children():
        where = f"{path}.{child.get_name()}" if path else child.get_name()
        # Lambdas and comprehensions see the enclosing function's locals.
        inherited = defined | {
            s.get_name() for s in table.get_symbols()
            if s.is_assigned() or s.is_imported() or s.is_parameter()
        }
        for sym in child.get_symbols():
            if sym.is_local() or sym.is_parameter() or sym.is_imported():
                continue
            if not sym.is_referenced():
                continue
            if sym.get_name() in inherited:
                continue
            problems.append(f"{where}: {sym.get_name()}")
        problems.extend(_walk(child, inherited, where))
    return problems


@pytest.mark.parametrize("path", _MODULES, ids=[p.name for p in _MODULES])
def test_no_function_references_an_undefined_name(path: Path) -> None:
    table = symtable.symtable(path.read_text(encoding="utf-8"),
                              str(path), "exec")
    problems = _walk(table, _module_level_names(table))
    assert not problems, (
        f"{path.name}: name(s) read but never bound — these raise NameError "
        f"only when the line executes:\n  " + "\n  ".join(problems)
    )
