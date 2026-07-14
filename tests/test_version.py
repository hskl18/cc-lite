from __future__ import annotations

import tomllib
from pathlib import Path

from cc_lite import __version__


def test_public_version_matches_authoritative_project_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["version"] == "0.2.0"
    assert __version__ == metadata["project"]["version"]
