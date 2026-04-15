import re
from typing import List, Dict
from groq import Groq
from sentence_transformers import SentenceTransformer
from agents.agent import Agent


class FrontierAgent(Agent):
    """
    Uses Groq (llama-3.3-70b-versatile) with RAG over the ChromaDB product vectorstore
    to estimate the fair market price of a product.

    Weights: 80% of the ensemble — the frontier model + RAG context is the strongest predictor.
    """

    name = "Frontier Agent"
    color = Agent.BLUE
    MODEL = "llama-3.3-70b-versatile"

    def __init__(self, collection):
        self.log("Initializing Frontier Agent")
        self.client = Groq()
        self.log("Frontier Agent is connected to Groq")
        self.collection = collection
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        message = "To provide context, here are similar items with known prices:\n\n"
        for similar, price in zip(similars, prices):
            message += f"Related product:\n{similar}\nPrice: ${price:.2f}\n\n"
        return message

    def messages_for(
        self, description: str, similars: List[str], prices: List[float]
    ) -> List[Dict[str, str]]:
        message = (
            f"Estimate the price of this product. "
            f"Respond with ONLY the price as a number — no dollar sign, no explanation.\n\n"
            f"{description}\n\n"
        )
        if similars:
            message += self.make_context(similars, prices)
        return [{"role": "user", "content": message}]

    def find_similars(self, description: str):
        self.log(
            "Frontier Agent is performing RAG search on ChromaDB to find 5 similar products"
        )
        count = self.collection.count()
        if count == 0:
            self.log("Frontier Agent: ChromaDB collection is empty — running without RAG context")
            return [], []
        vector = self.model.encode([description])
        results = self.collection.query(
            query_embeddings=vector.astype(float).tolist(), n_results=min(5, count)
        )
        documents = results["documents"][0][:]
        prices = [m["price"] for m in results["metadatas"][0][:]]
        self.log("Frontier Agent has found similar products")
        return documents, prices

    def get_price(self, s: str) -> float:
        s = s.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        documents, prices = self.find_similars(description)
        self.log(
            f"Frontier Agent is calling {self.MODEL} with 5 similar products as context"
        )
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=self.messages_for(description, documents, prices),
            seed=42,
            max_tokens=50,
        )
        reply = response.choices[0].message.content
        result = self.get_price(reply)
        self.log(f"Frontier Agent completed — predicting ${result:.2f}")
        return result
