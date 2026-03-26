"""Named Entity Recognition using transformers."""

from __future__ import annotations

from loguru import logger


def run_ner(text: str) -> list[dict]:
    """Run BERT-based NER on the given text.

    Returns a list of entity dicts with keys:
      ``entity``, ``score``, ``word``, ``start``, ``end``.
    """
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        pipeline,
    )

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    results = nlp(text)
    logger.info(f"NER found {len(results)} entities")
    return results
