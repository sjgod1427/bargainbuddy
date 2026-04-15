import json
from typing import Optional, List
from groq import Groq
from agents.agent import Agent
from agents.deals import Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent


class AutonomousPlanningAgent(Agent):
    """
    An advanced agentic planner that uses Groq's tool-calling capability to
    autonomously decide which deals to scan, value, and surface to the user.

    Unlike PlanningAgent (which follows a fixed pipeline), this agent reasons
    through the workflow itself — it can skip deals that clearly aren't good,
    focus on the most compelling ones, and decide when to notify.

    Use this as a drop-in replacement for PlanningAgent in framework.py.
    """

    name = "Autonomous Planning Agent"
    color = Agent.GREEN
    MODEL = "llama-3.3-70b-versatile"

    def __init__(self, collection):
        self.log("Autonomous Planning Agent is initializing")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.groq = Groq()
        self.memory = None
        self.opportunity = None
        self.log("Autonomous Planning Agent is ready")

    # ── Tool implementations ──────────────────────────────────────────────────

    def scan_the_internet_for_bargains(self) -> str:
        self.log("Autonomous Planning Agent is calling ScannerAgent")
        results = self.scanner.scan(memory=self.memory)
        return results.model_dump_json() if results else "No deals found"

    def estimate_true_value(self, description: str) -> str:
        self.log("Autonomous Planning Agent is estimating value via EnsembleAgent")
        estimate = self.ensemble.price(description)
        return f"The estimated true value of '{description}' is ${estimate:.2f}"

    def notify_user_of_deal(
        self,
        description: str,
        deal_price: float,
        estimated_true_value: float,
        url: str,
    ) -> str:
        if self.opportunity:
            self.log("Autonomous Planning Agent: second notify attempt ignored")
            return "Already notified once — ignoring duplicate notification."
        self.log("Autonomous Planning Agent is notifying the user")
        self.messenger.notify(description, deal_price, estimated_true_value, url)
        deal = Deal(product_description=description, price=deal_price, url=url)
        discount = estimated_true_value - deal_price
        self.opportunity = Opportunity(deal=deal, estimate=estimated_true_value, discount=discount)
        return "Notification sent successfully."

    # ── Tool schemas (OpenAI / Groq function-calling format) ─────────────────

    scan_function = {
        "name": "scan_the_internet_for_bargains",
        "description": "Returns top bargains scraped from the internet along with the price each item is being offered for.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }

    estimate_function = {
        "name": "estimate_true_value",
        "description": "Given the description of an item, estimate how much it is actually worth at retail.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the item to be estimated",
                },
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    }

    notify_function = {
        "name": "notify_user_of_deal",
        "description": "Send the user a push notification about the single most compelling deal. Call this only once.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Description of the item scraped from the internet",
                },
                "deal_price": {
                    "type": "number",
                    "description": "The price offered by this deal",
                },
                "estimated_true_value": {
                    "type": "number",
                    "description": "The estimated actual retail value",
                },
                "url": {
                    "type": "string",
                    "description": "The URL of the deal",
                },
            },
            "required": ["description", "deal_price", "estimated_true_value", "url"],
            "additionalProperties": False,
        },
    }

    def get_tools(self):
        return [
            {"type": "function", "function": self.scan_function},
            {"type": "function", "function": self.estimate_function},
            {"type": "function", "function": self.notify_function},
        ]

    def handle_tool_call(self, message):
        mapping = {
            "scan_the_internet_for_bargains": self.scan_the_internet_for_bargains,
            "estimate_true_value": self.estimate_true_value,
            "notify_user_of_deal": self.notify_user_of_deal,
        }
        results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = mapping.get(tool_name)
            result = tool(**arguments) if tool else "Unknown tool"
            results.append(
                {"role": "tool", "content": str(result), "tool_call_id": tool_call.id}
            )
        return results

    # ── Main agentic loop ─────────────────────────────────────────────────────

    SYSTEM_MESSAGE = (
        "You find great deals on bargain products using your tools and notify the user "
        "of the single best bargain you find."
    )
    USER_MESSAGE = (
        "First, use your tool to scan the internet for bargain deals. "
        "Then for each deal, use your tool to estimate its true value. "
        "Pick the single most compelling deal where the price is significantly lower "
        "than the estimated true value, and use your tool to notify the user. "
        "Then reply 'OK' to indicate success."
    )

    def plan(self, memory: List[str] = []) -> Optional[Opportunity]:
        self.log("Autonomous Planning Agent is kicking off a run")
        self.memory = memory
        self.opportunity = None

        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": self.USER_MESSAGE},
        ]

        done = False
        while not done:
            response = self.groq.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                tools=self.get_tools(),
            )
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                message = choice.message
                tool_results = self.handle_tool_call(message)

                # Serialize the assistant message to dict for the next API call
                msg_dict = {
                    "role": message.role,
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
                messages.append(msg_dict)
                messages.extend(tool_results)
            else:
                done = True

        reply = response.choices[0].message.content
        self.log(f"Autonomous Planning Agent completed with: {reply}")
        return self.opportunity
