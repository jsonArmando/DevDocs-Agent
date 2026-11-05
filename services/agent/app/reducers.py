"""Reducers del estado del grafo (puros, testeables sin LangGraph)."""
from __future__ import annotations

RESET = "__reset__"


def steps_reducer(old: list[str] | None, new: list[str] | None) -> list[str]:
    """Acumula los pasos del turno; el supervisor envía [RESET, ...] al iniciar cada turno.

    Necesario porque el checkpointer persiste el estado entre turnos del mismo thread_id.
    """
    new = new or []
    if new and new[0] == RESET:
        return list(new[1:])
    return (old or []) + list(new)
