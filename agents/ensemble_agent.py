from agents.agent import Agent
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.neural_network_agent import NeuralNetworkAgent
from agents.preprocessor import Preprocessor


class EnsembleAgent(Agent):
    """
    Orchestrates three pricing models and returns a weighted estimate.

    Weights (when neural network is available):
      - FrontierAgent (Groq llama-3.3-70b + RAG):  80%
      - SpecialistAgent (Groq llama-3.1-8b-instant): 10%
      - NeuralNetworkAgent (deep residual net):      10%

    If the NeuralNetworkAgent is unavailable (no weights file), weights
    are rebalanced to Frontier 89% + Specialist 11%.
    """

    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection):
        self.log("Initializing Ensemble Agent")
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.neural_network = NeuralNetworkAgent()
        self.preprocessor = Preprocessor()
        self.log("Ensemble Agent is ready")

    def price(self, description: str) -> float:
        self.log("Running Ensemble Agent — preprocessing text")
        rewrite = self.preprocessor.preprocess(description)
        self.log(f"Preprocessed with {self.preprocessor.model_name}")

        specialist = self.specialist.price(rewrite)
        frontier = self.frontier.price(rewrite)
        nn_result = self.neural_network.price(rewrite)

        if nn_result < 0:
            # Neural network unavailable — rebalance weights
            combined = frontier * 0.89 + specialist * 0.11
            self.log(
                f"Ensemble (no NN): Frontier=${frontier:.2f}, Specialist=${specialist:.2f} "
                f"→ combined=${combined:.2f}"
            )
        else:
            combined = frontier * 0.8 + specialist * 0.1 + nn_result * 0.1
            self.log(
                f"Ensemble: Frontier=${frontier:.2f}, Specialist=${specialist:.2f}, "
                f"NN=${nn_result:.2f} → combined=${combined:.2f}"
            )

        self.log(f"Ensemble Agent complete — returning ${combined:.2f}")
        return combined
