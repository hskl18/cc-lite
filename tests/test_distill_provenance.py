from __future__ import annotations

import sys
from pathlib import Path

from train.distill_engine import _engine_provenance


def test_engine_provenance_hashes_executable_and_file_arguments(tmp_path) -> None:
    engine_script = tmp_path / "engine.py"
    engine_script.write_text("print('fake engine')\n", encoding="utf-8")

    provenance = _engine_provenance(f'{sys.executable} "{engine_script}"')

    assert provenance["command"] == [sys.executable, str(engine_script)]
    assert provenance["executable"]["path"] == str(Path(sys.executable).resolve())
    assert len(provenance["executable"]["sha256"]) == 64
    assert provenance["argument_files"][0]["path"] == str(engine_script)
    assert len(provenance["argument_files"][0]["sha256"]) == 64
