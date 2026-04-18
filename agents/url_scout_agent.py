import json
import re
import requests
from bs4 import BeautifulSoup
from groq import Groq
from agents.agent import Agent


class URLScoutAgent(Agent):
    """
    Scrapes a product URL, estimates its true value via the EnsembleAgent,
    and returns a natural-language buy/skip recommendation.
    """

    name = "URL Scout Agent"
    color = Agent.CYAN
    MODEL = "llama-3.3-70b-versatile"

    VERDICT_PROMPT = """You are a savvy deal analyst. Given the product info below, write a short 3-4 sentence verdict telling the user whether they should buy it or skip it.

Product: {title}
Listed Price: ${listed_price:.2f}
Estimated True Market Value: ${estimated_value:.2f}
Discount vs Market Value: ${discount:.2f} ({discount_pct:.0f}%)
Description: {description}
URL: {url}

Be direct. Start with a clear BUY or SKIP recommendation, then explain why."""

    def __init__(self, ensemble):
        self.client = Groq()
        self.ensemble = ensemble

    def _scrape(self, url: str) -> dict:
        """Extract title, price, and description from a product page."""
        self.log(f"Scraping {url}")
        headers = {"User-Agent": "Mozilla/5.0 (compatible; BargainBuddy/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        title = self._extract_title(soup)
        price = self._extract_price(soup, resp.text)
        description = self._extract_description(soup)

        self.log(f"Scraped: title={title!r}, price={price}, desc_len={len(description)}")
        return {"title": title, "price": price, "description": description, "url": url}

    def _extract_title(self, soup: BeautifulSoup) -> str:
        for sel in [
            ("meta", {"property": "og:title"}),
            ("meta", {"name": "twitter:title"}),
        ]:
            tag = soup.find(sel[0], sel[1])
            if tag and tag.get("content"):
                return tag["content"].strip()
        if soup.title:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        return h1.get_text(strip=True) if h1 else "Unknown Product"

    def _extract_price(self, soup: BeautifulSoup, raw_html: str) -> float:
        # 1. JSON-LD structured data (most reliable)
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if item.get("@type") in ("Product", "Offer"):
                        offers = item.get("offers", item)
                        if isinstance(offers, dict):
                            p = offers.get("price") or offers.get("lowPrice")
                        elif isinstance(offers, list):
                            p = offers[0].get("price")
                        else:
                            p = None
                        if p:
                            return float(str(p).replace(",", ""))
            except Exception:
                pass

        # 2. Open Graph price meta tag
        tag = soup.find("meta", {"property": "product:price:amount"})
        if tag and tag.get("content"):
            try:
                return float(tag["content"].replace(",", ""))
            except ValueError:
                pass

        # 3. Regex scan for first $ price in HTML
        match = re.search(r'\$\s*([\d,]+(?:\.\d{2})?)', raw_html)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass

        return 0.0

    def _extract_description(self, soup: BeautifulSoup) -> str:
        for sel in [
            ("meta", {"property": "og:description"}),
            ("meta", {"name": "description"}),
            ("meta", {"name": "twitter:description"}),
        ]:
            tag = soup.find(sel[0], sel[1])
            if tag and tag.get("content"):
                return tag["content"].strip()[:500]
        # Fallback: first non-empty paragraph
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 60:
                return text[:500]
        return ""

    def analyse(self, url: str) -> str:
        """Scrape URL, estimate value, return a buy/skip verdict."""
        try:
            product = self._scrape(url)
        except Exception as e:
            return f"Sorry, I couldn't scrape that page: {e}"

        title = product["title"]
        listed_price = product["price"]
        description = product["description"]

        query = f"{title}. {description}"
        self.log("Running ensemble price estimation …")
        try:
            estimated_value = self.ensemble.price(query)
        except Exception as e:
            return f"Scraped the page but price estimation failed: {e}"

        if listed_price <= 0:
            return (
                f"**{title}**\n\n"
                f"I couldn't detect the listed price on this page automatically. "
                f"My model estimates a true market value of **${estimated_value:.2f}**. "
                f"If you can share the listed price, I can give you a full buy/skip verdict."
            )

        discount = estimated_value - listed_price
        discount_pct = (discount / estimated_value * 100) if estimated_value > 0 else 0

        self.log("Generating verdict with Groq …")
        prompt = self.VERDICT_PROMPT.format(
            title=title,
            listed_price=listed_price,
            estimated_value=estimated_value,
            discount=discount,
            discount_pct=discount_pct,
            description=description[:300],
            url=url,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            verdict = response.choices[0].message.content.strip()
        except Exception as e:
            verdict = (
                f"{'BUY' if discount > 20 else 'SKIP'} — "
                f"Listed at ${listed_price:.2f}, estimated value ${estimated_value:.2f} "
                f"(you {'save' if discount > 0 else 'overpay'} ${abs(discount):.2f})."
            )

        summary = (
            f"**{title}**\n\n"
            f"| | |\n|---|---|\n"
            f"| Listed Price | ${listed_price:.2f} |\n"
            f"| Est. Market Value | ${estimated_value:.2f} |\n"
            f"| Discount | ${discount:.2f} ({discount_pct:.0f}%) |\n\n"
            f"{verdict}"
        )
        return summary
