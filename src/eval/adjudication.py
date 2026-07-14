from __future__ import annotations


def material_adjudication(material_delta: int) -> float:
    if abs(material_delta) < 20:
        return 0.0
    return 1.0 if material_delta > 0 else -1.0
