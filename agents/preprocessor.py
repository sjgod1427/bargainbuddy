from litellm import completion
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# Default to Groq's fast model; override via env var for flexibility
DEFAULT_MODEL_NAME = os.getenv("PRICER_PREPROCESSOR_MODEL", "groq/llama-3.1-8b-instant")

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Preprocessor:
    """
    Rewrites raw product descriptions into a clean, structured format
    before passing them to the pricing models.

    Uses LiteLLM so the underlying model is easily swappable via the
    PRICER_PREPROCESSOR_MODEL environment variable.
    Default: groq/llama-3.1-8b-instant (fast and cheap)
    """

    def __init__(self, model_name=DEFAULT_MODEL_NAME, base_url=None):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.model_name = model_name
        self.base_url = base_url
        if "ollama" in model_name and not base_url:
            self.base_url = "http://localhost:11434"

    def messages_for(self, text: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

    def preprocess(self, text: str) -> str:
        messages = self.messages_for(text)
        response = completion(
            messages=messages,
            model=self.model_name,
            api_base=self.base_url,
        )
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        try:
            self.total_cost += response._hidden_params.get("response_cost", 0) or 0
        except Exception:
            pass
        return response.choices[0].message.content
