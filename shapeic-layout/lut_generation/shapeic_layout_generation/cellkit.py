"""Loading boundary between physical LUT generation and ShapeIC CellKit."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResolvedCellKit:
    """CellKit catalog plus the selected installed technology."""

    root: Path
    pdk_root: Path
    pdk: str
    catalog: Any
    technology: Any


def load_cellkit(root: Path, pdk_root: Path, pdk: str) -> ResolvedCellKit:
    """Load the ``shapeic_cellkit`` package from the selected source checkout."""

    source_root = root / "src"
    package_root = source_root / "shapeic_cellkit"
    if not package_root.is_dir():
        raise ValueError(
            f"cellkit root does not contain src/shapeic_cellkit: {root}"
        )
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)
    module = importlib.import_module("shapeic_cellkit")
    loaded_root = Path(module.__file__).resolve().parent
    if loaded_root != package_root.resolve():
        raise ValueError(
            "shapeic_cellkit is already loaded from a different checkout: "
            f"{loaded_root}"
        )
    catalog = module.CellKitCatalog.open(root, pdk, pdk_root)
    return ResolvedCellKit(
        root=root,
        pdk_root=pdk_root,
        pdk=pdk,
        catalog=catalog,
        technology=catalog.technology(),
    )
