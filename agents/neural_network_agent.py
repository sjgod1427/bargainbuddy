import os
from agents.agent import Agent
from agents.deep_neural_network import DeepNeuralNetworkInference


class NeuralNetworkAgent(Agent):
    """
    Runs the pre-trained deep residual neural network to estimate product prices.

    IMPORTANT: Requires 'deep_neural_network.pth' in the project root.
    Copy it from llm_engineering/week8/ (or week6/ where it was trained).
    If the weights file is missing, this agent gracefully returns -1 and the
    EnsembleAgent will reweight accordingly.

    Weights: 10% of the ensemble.
    """

    name = "Neural Network Agent"
    color = Agent.MAGENTA

    def __init__(self):
        self.log("Neural Network Agent is initializing")
        self.neural_network = DeepNeuralNetworkInference()
        self.available = False
        try:
            self.neural_network.setup()
            # Look for weights file in the project root
            model_path = "deep_neural_network.pth"
            if not os.path.exists(model_path):
                # Try one level up from agents/
                alt = os.path.join(os.path.dirname(__file__), "..", "deep_neural_network.pth")
                if os.path.exists(alt):
                    model_path = alt
            self.neural_network.load(model_path)
            self.available = True
            self.log("Neural Network Agent is ready — weights loaded")
        except FileNotFoundError:
            self.log(
                "Neural Network Agent: 'deep_neural_network.pth' not found. "
                "Copy it from week8/ or week6/ to enable this agent. "
                "Ensemble will reweight without it."
            )
        except Exception as e:
            self.log(f"Neural Network Agent failed to initialize: {e}. Running without it.")

    def price(self, description: str) -> float:
        """
        Estimate price using the neural network.
        Returns -1.0 if weights are not loaded (signal to EnsembleAgent to reweight).
        """
        if not self.available:
            self.log("Neural Network Agent is unavailable — returning sentinel -1")
            return -1.0
        self.log("Neural Network Agent is starting a prediction")
        result = self.neural_network.inference(description)
        self.log(f"Neural Network Agent completed — predicting ${result:.2f}")
        return result
