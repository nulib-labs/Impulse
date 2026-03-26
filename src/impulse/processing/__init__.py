"""Processing modules for the Impulse pipeline.

Modules are imported lazily to avoid pulling in heavy dependencies
(torch, spacy, marker-pdf, etc.) in lightweight Lambda/container contexts.
Import directly from the submodule you need, e.g.:

    from impulse.processing.images import process_image
"""
