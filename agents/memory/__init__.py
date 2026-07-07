"""Memory module: persistent context and global memory for the search process."""

from .record import MemRecord


def __getattr__(name):
    if name == "HybridRetriever":
        from .retriever import HybridRetriever

        return HybridRetriever
    if name == "GlobalMemoryLayer":
        from .global_memory import GlobalMemoryLayer

        return GlobalMemoryLayer
    raise AttributeError(name)

__all__ = [
    'HybridRetriever',
    'MemRecord',
    'GlobalMemoryLayer',
]
