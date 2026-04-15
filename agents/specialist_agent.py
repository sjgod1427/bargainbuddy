import re
from groq import Groq
from agents.agent import Agent


class SpecialistAgent(Agent):
    """
    A fast specialist price estimator powered by Groq's llama-3.1-8b-instant.

    In the original project this called a fine-tuned Llama model deployed on Modal.
    Here it uses Groq's fast inference API with a specialist pricing prompt instead,
    making the project fully self-contained without external Modal infrastructure.

    Weights: 10% of the ensemble.
    """

    name = "Specialist Agent"
    color = Agent.RED
    MODEL = "llama-3.1-8b-instant"

    SYSTEM_PROMPT = (
        "You are a specialist in estimating retail prices of consumer products. "
        "Given a product description, respond with ONLY a single number representing "
        "your best estimate of the current US retail price in dollars. "
        "No dollar sign. No explanation. No text. Just the number."
    )

    def __init__(self):
        self.log("Specialist Agent is initializing — connecting to Groq")
        self.client = Groq()
        self.log("Specialist Agent is ready")

    def get_price(self, s: str) -> float:
        s = s.replace("$", "").replace(",", "").strip()
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        self.log("Specialist Agent is calling Groq for a price estimate")
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Estimate the price of: {description}"},
            ],
            max_tokens=20,
        )
        result = self.get_price(response.choices[0].message.content)
        self.log(f"Specialist Agent completed — predicting ${result:.2f}")
        return result
