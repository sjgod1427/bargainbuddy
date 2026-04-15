"""
setup_vectorstore.py
====================
One-time setup: downloads a product dataset from HuggingFace, encodes each
product description with SentenceTransformers, and stores the results in the
local ChromaDB vectorstore that the FrontierAgent uses for RAG.

Run once before starting BargainBuddy:
    python setup_vectorstore.py

You can also copy the `products_vectorstore/` directory directly from
your llm_engineering/week8/ folder if you already have it populated.

Requirements: pip install -r requirements.txt + a valid HF_TOKEN in .env
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(override=True)

DB_PATH = "products_vectorstore"
COLLECTION_NAME = "products"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# HuggingFace dataset — adjust to whichever dataset you're using.
# The dataset must have columns: title, category, price (and optionally description/details).
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME", "sjgod1247/items_lite")
HF_SPLIT = os.getenv("HF_SPLIT", "train")
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "20000"))
BATCH_SIZE = 500

CATEGORIES = [
    "Appliances",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Musical_Instruments",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]


def build_document(row: dict) -> str:
    """Combine available fields into a single searchable document string."""
    parts = []
    if row.get("title"):
        parts.append(row["title"])
    if row.get("full"):
        parts.append(row["full"])
    elif row.get("description"):
        parts.append(row["description"])
    elif row.get("summary"):
        parts.append(row["summary"])
    if row.get("details"):
        parts.append(row["details"])
    return " ".join(parts).strip()


def main():
    print(f"Loading dataset: {HF_DATASET_NAME} (split={HF_SPLIT}, max={MAX_ITEMS})")
    ds = load_dataset(HF_DATASET_NAME, split=HF_SPLIT, token=os.getenv("HF_TOKEN"))
    ds = ds.select(range(min(MAX_ITEMS, len(ds))))
    print(f"Loaded {len(ds)} rows")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Connecting to ChromaDB at ./{DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    existing = collection.count()
    print(f"Collection already has {existing} items")

    rows = list(ds)
    total = len(rows)
    added = 0

    for start in tqdm(range(0, total, BATCH_SIZE), desc="Indexing batches"):
        batch = rows[start : start + BATCH_SIZE]
        documents, metadatas, embeddings, ids = [], [], [], []

        for i, row in enumerate(batch):
            doc = build_document(row)
            if not doc:
                continue
            price = float(row.get("price", 0) or 0)
            category = row.get("category", "Electronics")
            if category not in CATEGORIES:
                category = "Electronics"

            documents.append(doc)
            metadatas.append({"price": price, "category": category})
            ids.append(f"item_{start + i}")

        if not documents:
            continue

        vecs = model.encode(documents, show_progress_bar=False)
        collection.add(
            documents=documents,
            embeddings=vecs.tolist(),
            metadatas=metadatas,
            ids=ids,
        )
        added += len(documents)

    print(f"\nDone. Added {added} items. Collection now has {collection.count()} total.")


if __name__ == "__main__":
    main()
