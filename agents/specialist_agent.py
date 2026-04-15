import re
import os
import requests
from agents.agent import Agent


class SpecialistAgent(Agent):
    """
    Fine-tuned Llama-3.2-3B specialist price estimator — week8 architecture.

    Calls the Modal-hosted fine-tuned model endpoint (MODAL_ENDPOINT_URL in .env).
    Deploy the endpoint first with:  modal deploy modal_app.py

    Falls back to Groq llama-3.1-8b-instant if MODAL_ENDPOINT_URL is not set
    or the endpoint is unreachable.

    Weights: 10% of the ensemble.
    """

    name = "Specialist Agent"
    color = Agent.RED
    GROQ_MODEL = "llama-3.1-8b-instant"

    GROQ_SYSTEM_PROMPT = (
        "You are a specialist in estimating retail prices of consumer products. "
        "Given a product description, respond with ONLY a single number representing "
        "your best estimate of the current US retail price in dollars. "
        "No dollar sign. No explanation. No text. Just the number."
    )

    def __init__(self):
        self.modal_url = os.getenv("MODAL_ENDPOINT_URL", "").strip()
        self.groq_client = None
        if self.modal_url:
            self.log(f"Specialist Agent using Modal endpoint")
        else:
            self.log("MODAL_ENDPOINT_URL not set — using Groq fallback")
            from groq import Groq
            self.groq_client = Groq()
        self.log("Specialist Agent is ready")

    def get_price(self, s: str) -> float:
        s = s.replace("$", "").replace(",", "").strip()
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def _price_via_modal(self, description: str) -> float:
        self.log("Specialist Agent calling fine-tuned model on Modal")
        response = requests.post(
            self.modal_url,
            json={"description": description},
            timeout=300,  # allow up to 5 min for cold start
        )
        response.raise_for_status()
        return float(response.json()["price"])

    def _price_via_groq(self, description: str) -> float:
        if not self.groq_client:
            from groq import Groq
            self.groq_client = Groq()
        response = self.groq_client.chat.completions.create(
            model=self.GROQ_MODEL,
            messages=[
                {"role": "system", "content": self.GROQ_SYSTEM_PROMPT},
                {"role": "user", "content": f"Estimate the price of: {description}"},
            ],
            max_tokens=20,
        )
        return self.get_price(response.choices[0].message.content)

    def price(self, description: str) -> float:
        if self.modal_url:
            try:
                result = self._price_via_modal(description)
                self.log(f"Specialist Agent completed — predicting ${result:.2f}")
                return result
            except Exception as e:
                self.log(f"Modal endpoint failed ({e}) — falling back to Groq")

        self.log("Specialist Agent using Groq fallback")
        result = self._price_via_groq(description)
        self.log(f"Specialist Agent completed — predicting ${result:.2f}")
        return result
