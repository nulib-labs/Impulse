from fireworks.core.firework import FWAction, FireTaskBase
from typing import override

class NERTask(FireTaskBase):
    """
    Pulls all pages of a document from MongoDB document.
    Concatenates all text together.
    Runs bert-base-NER on the text together.
    Puts the data into nlp[works] coll

    """

    _fw_name = "NER Task"

    @override
    def run_task(self, fw_spec: dict) -> FWAction:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from transformers import pipeline

        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

        nlp = pipeline("ner", model=model, tokenizer=tokenizer)
        example = "My name is Wolfgang and I live in Berlin"

        ner_results = nlp(example)
        print(ner_results)
