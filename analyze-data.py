import os
import glob
import json
from pymongo import MongoClient
from bs4 import BeautifulSoup
from kristeva import Kristeva
from loguru import logger

# --- Setup ---
client = MongoClient(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING"))
db = client["fireworks"]
coll = db["filepad"]


# --- Helper function ---
def find_all_html(data):
    """Recursively find all 'html' values in nested dicts/lists."""
    results = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "html":
                results.append(value)
            else:
                results.extend(find_all_html(value))
    elif isinstance(data, list):
        for item in data:
            results.extend(find_all_html(item))
    return results


# --- Main ---
if __name__ == "__main__":
    json_files = glob.glob("./downloaded_files/**/*.json", recursive=True)
    logger.info(f"JSON files found: {json_files}")
    test_raw_documents = []

    for path in json_files:
        try:
            with open(path, "r") as f:
                data = json.load(f)

            html_fragments = find_all_html(data)
            combined_html = " ".join(html_fragments)
            cleantext = BeautifulSoup(combined_html, "lxml").get_text(
                separator=" ", strip=True
            )
            test_raw_documents.append(cleantext)

        except Exception as e:
            print(f"⚠️ Error processing {path}: {e}")

    # Save combined cleaned text to file
    with open("Cleaned.txt", "w") as f:
        for doc in test_raw_documents:
            f.write(doc + "\n\n")

    # Prepare data for Kristeva
    test_documents = {
        f"doc_{i}": [s.strip() for s in doc.split(". ") if s.strip()]
        for i, doc in enumerate(test_raw_documents)
    }

    # Train Kristeva model
    k = Kristeva(similarity_threshold=0.5, top_k=2)
    G = k.train(test_documents)
    fig = k.visualize_graph(G)
    fig.show()

    print(f"✅ Documents: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

    # Save trained model
    model_path = "kristeva_model.pkl"
    k.save_model(model_path)
    print(f"💾 Model saved to {model_path}")
