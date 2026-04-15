import json
from typing import Optional, List
from groq import Groq
from agents.deals import ScrapedDeal, DealSelection
from agents.agent import Agent


class ScannerAgent(Agent):
    """
    Scans RSS deal feeds and uses Groq (llama-3.3-70b-versatile) with JSON mode
    to select the 5 most promising deals with clear descriptions and prices.
    """

    MODEL = "llama-3.3-70b-versatile"

    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.

Respond strictly in JSON with no explanation, using EXACTLY this format:
{"deals": [{"product_description": "...", "price": 123.45, "url": "..."}]}

Rules:
- Provide the price as a number derived from the description.
- If the price of a deal isn't clear, do not include that deal.
- Select the 5 deals with the most detailed product descriptions and clear prices.
- Focus on the product itself, not the terms of the deal.
- Be careful with "$XXX off" or "reduced by $XXX" — that is NOT the actual price.
- Only include products when you are highly confident about the price.
"""

    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
Rephrase the description to be a summary of the product itself, not the terms of the deal.
Include a short paragraph of text in the product_description field for each of the 5 items you select.

Deals:

"""

    USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."

    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self):
        self.log("Scanner Agent is initializing")
        self.groq = Groq()
        self.log("Scanner Agent is ready")

    def fetch_deals(self, memory) -> List[ScrapedDeal]:
        self.log("Scanner Agent is about to fetch deals from RSS feed")
        urls = [opp.deal.url for opp in memory]
        scraped = ScrapedDeal.fetch()
        result = [scrape for scrape in scraped if scrape.url not in urls]
        self.log(f"Scanner Agent received {len(result)} deals not already scraped")
        return result

    def make_user_prompt(self, scraped) -> str:
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += "\n\n".join([scrape.describe() for scrape in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

    def scan(self, memory: List[str] = []) -> Optional[DealSelection]:
        """
        Call Groq with JSON mode to return a curated list of deals.
        """
        scraped = self.fetch_deals(memory)
        if scraped:
            user_prompt = self.make_user_prompt(scraped)
            self.log("Scanner Agent is calling Groq with JSON mode")
            response = self.groq.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=2000,
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            # Handle both {"deals": [...]} and direct list formats
            if isinstance(data, list):
                data = {"deals": data}
            result = DealSelection(**data)
            result.deals = [deal for deal in result.deals if deal.price > 0]
            self.log(
                f"Scanner Agent received {len(result.deals)} selected deals with price>0 from Groq"
            )
            return result
        return None

    def test_scan(self, memory: List[str] = []) -> Optional[DealSelection]:
        """
        Return a hardcoded DealSelection for testing without hitting the RSS feeds.
        """
        results = {
            "deals": [
                {
                    "product_description": "The Hisense R6 Series 55R6030N is a 55-inch 4K UHD Roku Smart TV featuring 3840x2160 resolution with Dolby Vision HDR and HDR10. It runs Roku OS with access to all major streaming services and supports voice control via Google Assistant and Alexa. Three HDMI ports make multi-device connection easy.",
                    "price": 178,
                    "url": "https://www.dealnews.com/products/Hisense/Hisense-R6-Series-55-R6030-N-55-4-K-UHD-Roku-Smart-TV/484824.html?iref=rss-c142",
                },
                {
                    "product_description": "The Poly Studio P21 is a 21.5-inch 1080p LED personal meeting display designed for video conferencing. It includes a 1080p webcam with manual pan/tilt/zoom, stereo speakers, a privacy shutter, ambient light sensor for vanity lighting adjustment, and 5W wireless charging.",
                    "price": 30,
                    "url": "https://www.dealnews.com/products/Poly-Studio-P21-21-5-1080-p-LED-Personal-Meeting-Display/378335.html?iref=rss-c39",
                },
                {
                    "product_description": "The Lenovo IdeaPad Slim 5 laptop features a 7th-gen AMD Ryzen 5 8645HS 6-core CPU, a 16-inch 1920x1080 touch display, 16GB RAM, and a 512GB SSD. Designed for efficient multitasking and everyday productivity tasks.",
                    "price": 446,
                    "url": "https://www.dealnews.com/products/Lenovo/Lenovo-Idea-Pad-Slim-5-7-th-Gen-Ryzen-5-16-Touch-Laptop/485068.html?iref=rss-c39",
                },
                {
                    "product_description": "The Dell G15 gaming laptop is powered by a 6th-gen AMD Ryzen 5 7640HS 6-Core CPU and Nvidia GeForce RTX 3050 GPU. It features a 15.6-inch 1080p display at 120Hz, 16GB RAM, and a 1TB NVMe SSD for smooth gaming and content creation.",
                    "price": 650,
                    "url": "https://www.dealnews.com/products/Dell/Dell-G15-Ryzen-5-15-6-Gaming-Laptop-w-Nvidia-RTX-3050/485067.html?iref=rss-c39",
                },
                {
                    "product_description": "The Apple AirPods Pro (2nd generation) feature active noise cancellation, Adaptive Transparency mode, and Personalized Spatial Audio. They include a MagSafe Charging Case with up to 30 hours total battery life and are sweat and water resistant (IPX4).",
                    "price": 189,
                    "url": "https://www.dealnews.com/products/Apple/Apple-AirPods-Pro-2nd-Gen/485100.html?iref=rss-c142",
                },
            ]
        }
        return DealSelection(**results)
