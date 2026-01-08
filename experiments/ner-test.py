from glob import glob
import spacy

# Load the English pipeline
nlp = spacy.load("en_core_web_trf")

for f in glob("./P0491_35556036056489/TXT/*.txt"):
    with open(f, "r") as g:
        text = g.readlines()
        text = " ".join(text).replace("\n", " ")
        doc = nlp(text)
        # Print named entities
        for ent in doc.ents:
            print(ent.text, ent.label_)

