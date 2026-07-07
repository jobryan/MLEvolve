"""Ablation controls and logging for MLEvolve memory mechanisms."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("MLEvolve")


def ablation_cfg(agent: Any) -> Any:
    return getattr(getattr(agent, "cfg", None), "ablation", None)


def child_memory_enabled(agent: Any) -> bool:
    return bool(getattr(ablation_cfg(agent), "child_memory", True))


def global_memory_filter(agent: Any) -> str:
    return str(getattr(ablation_cfg(agent), "global_memory_filter", "all") or "all")


def dissimilar_guidance_enabled(agent: Any) -> bool:
    return bool(getattr(ablation_cfg(agent), "dissimilar_guidance", False))


def _event_buffer(agent: Any) -> list[dict[str, Any]]:
    if not hasattr(agent, "current_memory_events"):
        agent.current_memory_events = []
    return agent.current_memory_events


def reset_memory_events(agent: Any) -> None:
    agent.current_memory_events = []


def record_memory_event(
    agent: Any,
    *,
    event_type: str,
    source: str,
    enabled: bool,
    records: list[dict[str, Any]] | None = None,
    text_chars: int = 0,
    reason: str | None = None,
) -> None:
    event = {
        "event_type": event_type,
        "source": source,
        "enabled": enabled,
        "record_count": len(records or []),
        "records": records or [],
        "text_chars": text_chars,
        "reason": reason,
    }
    _event_buffer(agent).append(event)


def get_child_memory(agent: Any, node: Any, *, include_code: bool = False, source: str = "child_history") -> str:
    if not child_memory_enabled(agent):
        record_memory_event(
            agent,
            event_type="child_memory",
            source=source,
            enabled=False,
            reason="ablation.child_memory=false",
        )
        return ""

    memory_text = node.fetch_child_memory(include_code=include_code)
    record_memory_event(
        agent,
        event_type="child_memory",
        source=source,
        enabled=True,
        records=[{"node_id": getattr(node, "id", None), "include_code": include_code}],
        text_chars=len(memory_text or ""),
    )
    return memory_text


def global_memory_available(agent: Any) -> bool:
    if global_memory_filter(agent) == "none":
        return False
    return bool(
        getattr(getattr(agent, "acfg", None), "use_global_memory", False)
        and getattr(agent, "global_memory", None) is not None
        and len(getattr(agent.global_memory, "records", [])) > 0
    )


def global_memory_recording_enabled(agent: Any) -> bool:
    if global_memory_filter(agent) == "none":
        return False
    return bool(getattr(getattr(agent, "acfg", None), "use_global_memory", False))


def _extract_stage(record: Any) -> str:
    title = getattr(record, "title", "") or ""
    if " - " in title:
        return title.split(" - ")[0]
    return "unknown"


def _record_payload(record: Any, score: float | None) -> dict[str, Any]:
    return {
        "record_id": getattr(record, "record_id", None),
        "title": getattr(record, "title", None),
        "label": getattr(record, "label", None),
        "stage": _extract_stage(record),
        "score": score,
    }


def _effective_label_filter(agent: Any, requested_label: int | None) -> int | None | str:
    mode = global_memory_filter(agent)
    if mode == "success_only":
        if requested_label is None or requested_label == 1:
            return 1
        return "no_compatible_records"
    if mode == "failure_only":
        if requested_label is None or requested_label == -1:
            return -1
        return "no_compatible_records"
    return requested_label


def retrieve_global_memory(
    agent: Any,
    *,
    query_text: str,
    top_k: int = 2,
    alpha: float = 0.5,
    dissimilar: bool = False,
    label_filter: int | None = None,
    stage_filter: str | None = None,
    min_score: float = 0.0,
    event_type: str = "global_memory_retrieval",
) -> list[tuple[Any, float]]:
    if not global_memory_available(agent):
        record_memory_event(
            agent,
            event_type=event_type,
            source="global_memory",
            enabled=False,
            reason=f"global_memory_filter={global_memory_filter(agent)} or memory unavailable",
        )
        return []

    effective_label = _effective_label_filter(agent, label_filter)
    if effective_label == "no_compatible_records":
        record_memory_event(
            agent,
            event_type=event_type,
            source="global_memory",
            enabled=True,
            reason=f"global_memory_filter={global_memory_filter(agent)} excludes label_filter={label_filter}",
        )
        return []

    results = agent.global_memory.retrieve_similar_records(
        query_text=query_text,
        top_k=top_k,
        alpha=alpha,
        dissimilar=dissimilar,
        label_filter=effective_label,
        stage_filter=stage_filter,
        min_score=min_score,
    )
    record_memory_event(
        agent,
        event_type=event_type,
        source="global_memory",
        enabled=True,
        records=[_record_payload(record, score) for record, score in results],
        reason=f"global_memory_filter={global_memory_filter(agent)}",
    )
    logger.info(
        "[MemoryAblation] %s retrieved %d records (mode=%s, dissimilar=%s, label_filter=%s, stage_filter=%s)",
        event_type,
        len(results),
        global_memory_filter(agent),
        dissimilar,
        effective_label,
        stage_filter,
    )
    return results
