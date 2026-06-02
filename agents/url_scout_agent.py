import json
import re
import subprocess
import requests
from bs4 import BeautifulSoup
from groq import Groq
from agents.agent import Agent

# Realistic browser headers — Amazon and most e-commerce sites check these
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Currency symbols → ISO code
CURRENCY_MAP = {
    "$": "USD", "₹": "INR", "€": "EUR", "£": "GBP",
    "¥": "JPY", "₩": "KRW", "A$": "AUD", "C$": "CAD",
}


def _ensure_playwright_browsers():
    """Install Playwright Chromium if not already present (runs once at startup)."""
    try:
        result = subprocess.run(
            ["playwright", "install", "chromium", "--with-deps"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            print("Playwright: Chromium ready.")
        else:
            print(f"Playwright install warning: {result.stderr[:200]}")
    except Exception as e:
        print(f"Playwright install skipped: {e}")


class URLScoutAgent(Agent):
    """
    Scrapes a product URL, estimates its true value via the EnsembleAgent,
    and returns a natural-language buy/skip recommendation.

    Scraping strategy (in order):
      1. requests + BeautifulSoup with realistic browser headers
      2. Playwright headless Chromium (handles JS redirects and JS-rendered prices)
    """

    name = "URL Scout Agent"
    color = Agent.CYAN
    MODEL = "llama-3.3-70b-versatile"

    VERDICT_PROMPT = """You are a savvy deal analyst. Given the product info below, write a short 3-4 sentence verdict telling the user whether they should buy it or skip it.

Product: {title}
Listed Price: {listed_price_display}
Estimated True Market Value (USD): ${estimated_value:.2f}
Description: {description}
URL: {url}

{currency_note}

Be direct. Start with a clear BUY or SKIP recommendation, then explain why in plain language."""

    def __init__(self, ensemble):
        self.client = Groq()
        self.ensemble = ensemble

    # ── Price / title / description extractors ─────────────────────────────

    def _extract_title(self, soup: BeautifulSoup) -> str:
        for tag, attr in [
            ("meta", {"property": "og:title"}),
            ("meta", {"name": "twitter:title"}),
        ]:
            el = soup.find(tag, attr)
            if el and el.get("content"):
                return el["content"].strip()
        # Amazon-specific
        for selector in ["#productTitle", "#title", "h1.product-title"]:
            el = soup.select_one(selector)
            if el:
                return el.get_text(strip=True)
        if soup.title:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        return h1.get_text(strip=True) if h1 else "Unknown Product"

    def _extract_price(self, soup: BeautifulSoup, raw_html: str) -> tuple[float, str]:
        """Return (price_as_float, currency_symbol). Price is always the raw number."""

        # 1. JSON-LD structured data
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if item.get("@type") in ("Product", "Offer"):
                        offers = item.get("offers", item)
                        if isinstance(offers, list):
                            offers = offers[0]
                        p = offers.get("price") or offers.get("lowPrice") if isinstance(offers, dict) else None
                        if p:
                            currency = offers.get("priceCurrency", "")
                            sym = next((s for s, c in CURRENCY_MAP.items() if c == currency), "$")
                            return float(str(p).replace(",", "")), sym
            except Exception:
                pass

        # 2. Amazon-specific DOM elements (handles .a-price-whole + .a-price-fraction)
        whole = soup.select_one(".a-price-whole")
        frac = soup.select_one(".a-price-fraction")
        if whole:
            price_str = whole.get_text(strip=True).replace(",", "").rstrip(".")
            if frac:
                price_str += "." + frac.get_text(strip=True)
            try:
                return float(price_str), "₹" if "amazon.in" in raw_html[:500] else "$"
            except ValueError:
                pass

        # 3. Open Graph / meta price tag
        for prop in ["product:price:amount", "og:price:amount"]:
            tag = soup.find("meta", {"property": prop})
            if tag and tag.get("content"):
                try:
                    return float(tag["content"].replace(",", "")), "$"
                except ValueError:
                    pass

        # 4. Regex — matches $, ₹, €, £ followed by a number
        match = re.search(r'([₹$€£])\s*([\d,]+(?:\.\d{1,2})?)', raw_html)
        if match:
            sym = match.group(1)
            try:
                return float(match.group(2).replace(",", "")), sym
            except ValueError:
                pass

        return 0.0, "$"

    def _extract_description(self, soup: BeautifulSoup) -> str:
        for tag, attr in [
            ("meta", {"property": "og:description"}),
            ("meta", {"name": "description"}),
            ("meta", {"name": "twitter:description"}),
        ]:
            el = soup.find(tag, attr)
            if el and el.get("content"):
                return el["content"].strip()[:500]
        # Amazon feature bullets
        bullets = soup.select("#feature-bullets li span.a-list-item")
        if bullets:
            return " ".join(b.get_text(strip=True) for b in bullets[:5])[:500]
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 60:
                return text[:500]
        return ""

    # ── Two-tier scraping ───────────────────────────────────────────────────

    def _scrape_with_requests(self, url: str) -> dict | None:
        """Fast path: requests + BeautifulSoup. Returns None if price/title missing."""
        session = requests.Session()
        resp = session.get(url, headers=BROWSER_HEADERS, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        title = self._extract_title(soup)
        price, currency = self._extract_price(soup, resp.text)
        description = self._extract_description(soup)
        if price > 0 and title != "Unknown Product":
            return {"title": title, "price": price, "currency": currency,
                    "description": description, "url": resp.url}
        return None

    def _scrape_with_playwright(self, url: str) -> dict:
        """Fallback: headless Chromium via Playwright for JS-rendered pages."""
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent=BROWSER_HEADERS["User-Agent"],
                locale="en-US",
            )
            page = ctx.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(3000)  # let JS finish rendering
            final_url = page.url
            content = page.content()
            browser.close()
        soup = BeautifulSoup(content, "html.parser")
        title = self._extract_title(soup)
        price, currency = self._extract_price(soup, content)
        description = self._extract_description(soup)
        return {"title": title, "price": price, "currency": currency,
                "description": description, "url": final_url}

    def _scrape(self, url: str) -> dict:
        self.log(f"Scraping {url}")

        # Try fast path first
        try:
            result = self._scrape_with_requests(url)
            if result:
                self.log(f"Requests scrape OK: {result['title']!r}, price={result['price']} {result['currency']}")
                return result
            self.log("Requests scrape returned no price/title — trying Playwright")
        except Exception as e:
            self.log(f"Requests failed ({e}) — trying Playwright")

        # Playwright fallback
        self.log("Launching headless Chromium via Playwright")
        result = self._scrape_with_playwright(url)
        self.log(f"Playwright scrape: {result['title']!r}, price={result['price']} {result['currency']}")
        return result

    # ── Public interface ────────────────────────────────────────────────────

    def analyse(self, url: str) -> str:
        """Scrape URL, estimate value, return a buy/skip verdict."""
        try:
            product = self._scrape(url)
        except Exception as e:
            return f"Sorry, I couldn't scrape that page: {e}"

        title = product["title"]
        listed_price = product["price"]
        currency = product.get("currency", "$")
        description = product["description"]
        final_url = product["url"]

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
                f"My model estimates a true market value of **${estimated_value:.2f} USD**. "
                f"If you can share the listed price, I can give you a full buy/skip verdict."
            )

        # Format price display with correct currency symbol
        listed_price_display = f"{currency}{listed_price:,.2f}"

        # Currency note for the LLM if non-USD
        currency_name = CURRENCY_MAP.get(currency, currency)
        if currency != "$":
            currency_note = (
                f"Note: the listed price is in {currency_name} ({currency}). "
                f"The estimated market value is in USD. Factor in the exchange rate when comparing."
            )
        else:
            currency_note = ""

        discount_usd = estimated_value - listed_price  # only meaningful if both USD
        discount_pct = (discount_usd / estimated_value * 100) if estimated_value > 0 else 0

        self.log("Generating verdict with Groq …")
        prompt = self.VERDICT_PROMPT.format(
            title=title,
            listed_price_display=listed_price_display,
            estimated_value=estimated_value,
            description=description[:300],
            url=final_url,
            currency_note=currency_note,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
            )
            verdict = response.choices[0].message.content.strip()
        except Exception as e:
            verdict = f"Listed at {listed_price_display}, estimated USD value ${estimated_value:.2f}."

        rows = f"| Listed Price | {listed_price_display} |\n| Est. Market Value (USD) | ${estimated_value:.2f} |"
        if currency == "$":
            rows += f"\n| Discount | ${discount_usd:.2f} ({discount_pct:.0f}%) |"

        summary = (
            f"**{title}**\n\n"
            f"| | |\n|---|---|\n"
            f"{rows}\n\n"
            f"{verdict}"
        )
        return summary
