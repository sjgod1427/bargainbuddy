---
title: BargainBuddy
emoji: 🛍️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.25.2"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# BargainBuddy

**Live demo → [huggingface.co/spaces/sjgod1247/bargainBuddy](https://huggingface.co/spaces/sjgod1247/bargainBuddy)**

An autonomous multi-agent deal-hunting system that scans online deal feeds, estimates the true market value of products using an ensemble of AI models, and sends push notifications when it finds a genuine bargain.

Powered entirely by **Groq** (fast open-source LLMs) — no OpenAI required.

---

## Features

- **Autonomous deal scanner** — scrapes RSS deal feeds every 5 minutes, picks the best 5 deals using Groq JSON-mode
- **Ensemble price estimator** — combines a RAG frontier model, specialist model, and PyTorch neural network to estimate true market value
- **Live agent log** — colour-coded real-time output from every agent
- **3D vectorstore plot** — interactive scatter plot of 20K+ product embeddings coloured by category
- **URL deal checker** — paste any product URL and get an instant BUY / SKIP verdict with price comparison
- **Per-user push notifications** — each visitor can enter their own Pushover credentials to get alerts on their phone

---

## Architecture

```
RSS Feeds
    │
    ▼
┌─────────────────┐   picks 5 best deals (JSON mode)
│  ScannerAgent   │──────────────────────────────────────────────────────────┐
│  llama-3.3-70b  │                                                          │
└─────────────────┘                                                          │
                                                                             ▼
                                                               ┌─────────────────────┐
                                                               │   EnsembleAgent     │
                                                               │ ┌─────────────────┐ │
                                                               │ │  FrontierAgent  │ │  80%
                                                               │ │  llama-3.3-70b  │ │
                                                               │ │  + ChromaDB RAG │ │
                                                               │ └─────────────────┘ │
                                                               │ ┌─────────────────┐ │
                                                               │ │SpecialistAgent  │ │  10%
                                                               │ │llama-3.1-8b-ins │ │
                                                               │ └─────────────────┘ │
                                                               │ ┌─────────────────┐ │
                                                               │ │  NeuralNetwork  │ │  10%
                                                               │ │  Agent (PyTorch)│ │
                                                               │ └─────────────────┘ │
                                                               └────────┬────────────┘
                                                                        │ weighted price estimate
                                                                        ▼
                                                        ┌───────────────────────────┐
                                                        │       PlanningAgent       │
                                                        │  if discount > $50 →      │
                                                        │     MessagingAgent        │
                                                        │     (llama-3.3-70b)       │
                                                        │     → Pushover push       │
                                                        └───────────────────────────┘

User pastes URL
    │
    ▼
┌─────────────────┐   scrape title, price, description
│  URLScoutAgent  │──────────────────────────────────────────────────────────┐
│  BeautifulSoup  │                                                          │
│  + JSON-LD      │                                                          ▼
└─────────────────┘                                               EnsembleAgent (price)
                                                                             │
                                                                             ▼
                                                                  Groq llama-3.3-70b
                                                                  BUY / SKIP verdict
```

### Agents

| Agent | Model | Role |
|-------|-------|------|
| **ScannerAgent** | `llama-3.3-70b-versatile` | Reads deal RSS feeds; picks 5 best via JSON-mode structured output |
| **FrontierAgent** | `llama-3.3-70b-versatile` | RAG over 20K+ product embeddings in ChromaDB; gives price estimate with context |
| **SpecialistAgent** | `llama-3.1-8b-instant` | Fast specialist price estimator |
| **NeuralNetworkAgent** | PyTorch (deep residual net) | Local neural network prediction (requires `deep_neural_network.pth`) |
| **EnsembleAgent** | — | Combines the three price estimates with weighted averaging |
| **PlanningAgent** | — | Orchestrates the pipeline; triggers alerts on large discounts |
| **MessagingAgent** | `llama-3.3-70b-versatile` | Crafts an exciting notification; delivers via Pushover API |
| **URLScoutAgent** | `llama-3.3-70b-versatile` | Scrapes any product URL and gives a buy/skip verdict |

---

## Quick Start (Local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — GROQ_API_KEY is required
# PUSHOVER_USER + PUSHOVER_TOKEN are optional (for push notifications)
```

### 3. Run

```bash
python app.py
```

The vectorstore is included in the repo — no setup step needed.

---

## Push Notifications

Sign up free at [pushover.net](https://pushover.net), then either:

- **Self-hosted / local**: add `PUSHOVER_USER` and `PUSHOVER_TOKEN` to your `.env`
- **HuggingFace Spaces**: open the **🔔 Push Notifications** panel in the UI and enter your own keys — they stay in your browser session only

Notifications fire automatically when a deal with a discount > $50 is found, or manually by clicking any row in the deals table.

---

## Project Structure

```
bargainbuddy/
├── agents/
│   ├── agent.py                      # Abstract base with colored logging
│   ├── deals.py                      # Data models: ScrapedDeal, Deal, DealSelection, Opportunity
│   ├── scanner_agent.py              # RSS scraper + Groq JSON-mode deal selector
│   ├── frontier_agent.py             # Groq + ChromaDB RAG price estimator
│   ├── specialist_agent.py           # Groq fast specialist price estimator
│   ├── neural_network_agent.py       # PyTorch deep residual net price estimator
│   ├── ensemble_agent.py             # Weighted combination of the three models
│   ├── planning_agent.py             # Main pipeline orchestrator
│   ├── autonomous_planning_agent.py  # Alternative: autonomous tool-calling planner
│   ├── messaging_agent.py            # Groq notification writer + Pushover push
│   ├── url_scout_agent.py            # Scrapes product URLs + buy/skip verdict
│   ├── preprocessor.py               # LiteLLM-based text normalizer
│   ├── deep_neural_network.py        # PyTorch model architecture + inference
│   ├── items.py                      # Item dataclass for evaluation datasets
│   └── evaluator.py                  # Tester class: MSE / R² / scatter charts
├── products_vectorstore/             # Pre-built ChromaDB (20K product embeddings)
├── framework.py                      # DealAgentFramework: ChromaDB + memory + orchestration
├── app.py                            # Gradio web UI
├── log_utils.py                      # ANSI → HTML color converter for Gradio logs
├── setup_vectorstore.py              # One-time ChromaDB population from HuggingFace
├── requirements.txt
├── .env.example
└── README.md
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ | API key from [console.groq.com](https://console.groq.com) |
| `PUSHOVER_USER` | optional | Pushover user key for push notifications |
| `PUSHOVER_TOKEN` | optional | Pushover app token |
| `HF_TOKEN` | optional | HuggingFace token (only needed to rebuild vectorstore) |
| `HF_DATASET_NAME` | optional | Dataset to index (default: `sjgod1247/items_lite`) |
| `MAX_ITEMS` | optional | Max products to index (default: `20000`) |
| `MODAL_ENDPOINT_URL` | optional | Fine-tuned specialist endpoint (falls back to Groq if unset) |
