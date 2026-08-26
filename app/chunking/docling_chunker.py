from typing import Iterable
from docling.chunking import HierarchicalChunker,HybridChunker

def hierarchical_chunks(document) ->Iterable:
    """
    Generate chunks using Docling's document hierarchy.
    """
    chunker = HierarchicalChunker()
    return chunker.chunk(document)

def hybrid_chunks(document)-> Iterable:
    """
    Generate chunks using Docling's hierarchical + token-aware hybrid strategy.
    """
    chunker = HybridChunker()
    return list(chunker.chunk(document))