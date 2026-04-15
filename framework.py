"""
BargainBuddy Framework
======================
The central runtime that wires together the agent pipeline, manages the
ChromaDB vector store, and persists discovered opportunities to disk.

Usage:
    python framework.py          # single run in the terminal
    python app.py                # launch the Gradio web UI
"""

import os
import sys
import logging
import json
from typing import List
from dotenv import load_dotenv
import chromadb
from agents.planning_agent import PlanningAgent
from agents.deals import Opportunity
from sklearn.manifold import TSNE
import numpy as np

load_dotenv(override=True)

# Logging colors
BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

# Product categories and their plot colors
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
COLORS = ["red", "blue", "brown", "orange", "yellow", "green", "purple", "cyan"]


def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


class DealAgentFramework:
    """
    Top-level orchestrator for BargainBuddy.

    - Connects to the ChromaDB product vectorstore
    - Loads / persists deal memory (memory.json)
    - Delegates each run to the PlanningAgent pipeline
    - Exposes plot data for the Gradio 3D visualization
    """

    DB = "products_vectorstore"
    MEMORY_FILENAME = "memory.json"

    def __init__(self):
        init_logging()
        client = chromadb.PersistentClient(path=self.DB)
        self.memory = self.read_memory()
        self.collection = client.get_or_create_collection("products")
        self.planner = None

    def init_agents_as_needed(self):
        if not self.planner:
            self.log("Initializing Agent Framework")
            self.planner = PlanningAgent(self.collection)
            self.log("Agent Framework is ready")

    def read_memory(self) -> List[Opportunity]:
        if os.path.exists(self.MEMORY_FILENAME):
            with open(self.MEMORY_FILENAME, "r") as f:
                data = json.load(f)
            return [Opportunity(**item) for item in data]
        return []

    def write_memory(self) -> None:
        data = [opp.model_dump() for opp in self.memory]
        with open(self.MEMORY_FILENAME, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def reset_memory(cls) -> None:
        """Trim memory down to the 2 most recent entries (for testing)."""
        data = []
        if os.path.exists(cls.MEMORY_FILENAME):
            with open(cls.MEMORY_FILENAME, "r") as f:
                data = json.load(f)
        with open(cls.MEMORY_FILENAME, "w") as f:
            json.dump(data[:2], f, indent=2)

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[BargainBuddy Framework] " + message + RESET
        logging.info(text)

    def run(self) -> List[Opportunity]:
        self.init_agents_as_needed()
        logging.info("Kicking off Planning Agent")
        result = self.planner.plan(memory=self.memory)
        logging.info(f"Planning Agent completed — result: {result}")
        if result:
            self.memory.append(result)
            self.write_memory()
        return self.memory

    @classmethod
    def get_plot_data(cls, max_datapoints: int = 2000):
        """Return (documents, 3-D t-SNE vectors, colors) for the product database visualization."""
        client = chromadb.PersistentClient(path=cls.DB)
        collection = client.get_or_create_collection("products")
        result = collection.get(
            include=["embeddings", "documents", "metadatas"],
            limit=max_datapoints,
        )
        if result["embeddings"] is None or len(result["embeddings"]) == 0:
            return [], np.array([]).reshape(0, 3), []
        vectors = np.array(result["embeddings"])
        documents = result["documents"]
        categories = [metadata.get("category", "Electronics") for metadata in result["metadatas"]]
        colors = [
            COLORS[CATEGORIES.index(c)] if c in CATEGORIES else "grey"
            for c in categories
        ]
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced_vectors = tsne.fit_transform(vectors)
        return documents, reduced_vectors, colors


if __name__ == "__main__":
    DealAgentFramework().run()
