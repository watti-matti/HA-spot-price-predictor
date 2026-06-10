"""Lint guard for the shipped example Lovelace/ApexCharts dashboards.

These dashboards are copy-pasted by users, so a broken `data_generator`
ships as a support ticket. This test encodes the apexcharts-card gotchas
that bit us repeatedly during the v2.11 cleanup, so they can't regress
into a release:

1. Attributes must be read via ``entity.attributes.<name>`` — never
   ``entity.<name>`` (the latter is undefined → empty chart).
2. ``data_generator`` must return ``[x, y]`` arrays — never ``{x, y}``
   objects (apexcharts-card renders nothing for objects).
3. apexcharts-card is time-series only: ``xaxis.type: category`` does not
   render index/k axes; such charts must map the index onto timestamps.
4. DtACI is FI-only since v2.11.8 — no dashboard may loop over the removed
   ``se1`` / ``se3`` / ``ee`` calibration zones.

It also asserts every example file is valid YAML.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DASHBOARDS = sorted(
    [*(REPO / "docs" / "yaml_examples").glob("*.yaml"),
     *( [REPO / "ha_dashboard.yaml"] if (REPO / "ha_dashboard.yaml").exists() else [])]
)


def _ids(paths):
    return [p.relative_to(REPO).as_posix() for p in paths]


@pytest.mark.parametrize("path", DASHBOARDS, ids=_ids(DASHBOARDS))
def test_dashboard_is_valid_yaml(path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    list(yaml.safe_load_all(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("path", DASHBOARDS, ids=_ids(DASHBOARDS))
def test_no_bare_entity_attribute_access(path: Path) -> None:
    """`entity.<attr>` without `.attributes.` is undefined in apexcharts-card."""
    text = path.read_text(encoding="utf-8")
    bad = re.findall(
        r"entity\.(?!attributes\.)"
        r"(daily_forecast|forecast|dtaci_diagnostics|timeline|week_[a-z_]+)",
        text,
    )
    assert not bad, (
        f"{path.name}: use `entity.attributes.<name>` not `entity.<name>` "
        f"for: {sorted(set(bad))}")


@pytest.mark.parametrize("path", DASHBOARDS, ids=_ids(DASHBOARDS))
def test_data_generators_return_arrays_not_objects(path: Path) -> None:
    """`data_generator` must return [x, y] arrays; `=> ({ ... })` (object
    points) render nothing in apexcharts-card."""
    text = path.read_text(encoding="utf-8")
    # Arrow function returning an object literal as a chart point.
    matches = re.findall(r"=>\s*\(\{", text)
    assert not matches, (
        f"{path.name}: {len(matches)} data point(s) returned as {{x,y}} "
        f"objects via `=> ({{`; return `[x, y]` arrays instead")


@pytest.mark.parametrize("path", DASHBOARDS, ids=_ids(DASHBOARDS))
def test_no_category_xaxis(path: Path) -> None:
    """apexcharts-card is time-series only — a category x-axis does not
    render. Index/k axes must be mapped onto timestamps."""
    text = path.read_text(encoding="utf-8")
    assert "type: category" not in text, (
        f"{path.name}: `xaxis.type: category` does not render in "
        f"apexcharts-card; map the index onto timestamps and relabel")


@pytest.mark.parametrize("path", DASHBOARDS, ids=_ids(DASHBOARDS))
def test_no_removed_dtaci_neighbour_zones(path: Path) -> None:
    """DtACI is FI-only since v2.11.8; the se1/se3/ee zone bundles are gone."""
    text = path.read_text(encoding="utf-8")
    for zone in ("'se1'", '"se1"', "'se3'", '"se3"'):
        assert zone not in text, (
            f"{path.name}: references removed DtACI zone {zone} "
            f"(FI-only since v2.11.8)")
