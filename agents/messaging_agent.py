import os
from agents.deals import Opportunity
from agents.agent import Agent
from groq import Groq
import requests

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


class MessagingAgent(Agent):
    """
    Crafts compelling deal notifications using Groq (via LiteLLM) and
    delivers them as push notifications via the Pushover API.
    """

    name = "Messaging Agent"
    color = Agent.WHITE
    MODEL = "llama-3.1-8b-instant"

    def __init__(self):
        self.log("Messaging Agent is initializing")
        self.client = Groq()
        self.pushover_user = os.getenv("PUSHOVER_USER", "")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN", "")
        self.log("Messaging Agent has initialized Pushover and Groq")

    def push(self, text: str):
        """Send a push notification via Pushover."""
        if not self.pushover_user or not self.pushover_token:
            self.log("Messaging Agent: Pushover credentials not set — skipping push")
            return
        self.log("Messaging Agent is sending a push notification")
        payload = {
            "user": self.pushover_user,
            "token": self.pushover_token,
            "message": text,
            "sound": "cashregister",
        }
        requests.post(PUSHOVER_URL, data=payload)

    def alert(self, opportunity: Opportunity):
        """Send a structured alert about the given Opportunity."""
        text = f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
        text += f"Estimate=${opportunity.estimate:.2f}, "
        text += f"Discount=${opportunity.discount:.2f}: "
        text += opportunity.deal.product_description[:80] + "... "
        text += opportunity.deal.url
        self.push(text)
        self.log("Messaging Agent has completed alert")

    def craft_message(
        self, description: str, deal_price: float, estimated_true_value: float
    ) -> str:
        """Use Groq to write an exciting 2-3 sentence push notification."""
        user_prompt = (
            "Please summarize this great deal in 2-3 sentences for an exciting push notification.\n"
            f"Item Description: {description}\n"
            f"Offered Price: ${deal_price:.2f}\n"
            f"Estimated true value: ${estimated_true_value:.2f}\n\n"
            "Respond ONLY with the 2-3 sentence message. Make it exciting and concise."
        )
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content

    def notify(
        self, description: str, deal_price: float, estimated_true_value: float, url: str
    ):
        """Craft a message with Groq and push it to the user."""
        self.log("Messaging Agent is using Groq to craft the notification message")
        try:
            text = self.craft_message(description, deal_price, estimated_true_value)
            message = text[:200] + "... " + url
        except Exception as e:
            self.log(f"Messaging Agent could not craft message ({e}) — using fallback")
            discount = estimated_true_value - deal_price
            message = (
                f"Deal Alert! {description[:80]}... "
                f"Price: ${deal_price:.2f} | Est. value: ${estimated_true_value:.2f} | "
                f"You save: ${discount:.2f} {url}"
            )
        self.push(message)
        self.log("Messaging Agent has completed notification")
