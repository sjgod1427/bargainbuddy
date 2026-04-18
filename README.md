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

An autonomous multi-agent deal-hunting system that scans online deal feeds, estimates the true market value of products using an ensemble of AI models, and sends you push notifications when it finds a genuine bargain.

Powered entirely by **Groq** (fast open-source LLMs) — no OpenAI required.

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
                                                        │     PlanningAgent         │
                                                        │  if discount > $50 →      │
                                                        │     MessagingAgent        │
                                                        │     (llama-3.3-70b)       │
                                                        │     → Pushover push       │
                                                        └───────────────────────────┘
```

### Agents

| Agent | Model | Role |
|-------|-------|------|
| **ScannerAgent** | `llama-3.3-70b-versatile` | Reads deal RSS feeds; picks 5 best via JSON-mode structured output |
| **FrontierAgent** | `llama-3.3-70b-versatile` | RAG over 50K+ product embeddings in ChromaDB; gives price estimate with context |
| **SpecialistAgent** | `llama-3.1-8b-instant` | Fast specialist price estimate (replaces the original Modal fine-tuned model) |
| **NeuralNetworkAgent** | PyTorch (deep residual net) | Local neural network prediction (requires `deep_neural_network.pth`) |
| **EnsembleAgent** | — | Combines the three price estimates with weighted averaging |
| **PlanningAgent** | — | Orchestrates the pipeline; triggers alerts on large discounts |
| **MessagingAgent** | `llama-3.3-70b-versatile` | Crafts an exciting notification; delivers via Pushover API |
| **AutonomousPlanningAgent** | `llama-3.3-70b-versatile` | Alternative to PlanningAgent — uses Groq tool-calling to reason autonomously |

---

## Quick Start

### 1. Install dependencies

```bash
cd bargainbuddy
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (required)
# Add PUSHOVER_USER and PUSHOVER_TOKEN if you want push notifications
```

### 3. Set up the product vectorstore

**Option A — Copy from week8 (fastest):**
```bash
cp -r ../llm_engineering/week8/products_vectorstore ./products_vectorstore
```

**Option B — Download and index from HuggingFace:**
```bash
python setup_vectorstore.py
```

### 4. (Optional) Copy neural network weights

```bash
cp ../llm_engineering/week8/deep_neural_network.pth ./deep_neural_network.pth
# or from week6 where it was trained
```

If this file is missing, BargainBuddy runs fine — the EnsembleAgent
rebalances to Frontier 89% + Specialist 11%.

### 5. Run

**CLI (single run):**
```bash
python framework.py
```

**Web UI (Gradio — auto-refreshes every 5 min):**
```bash
python app.py
```

---

## Project Structure

```
bargainbuddy/
├── agents/
│   ├── agent.py                  # Abstract base with colored logging
│   ├── deals.py                  # Data models: ScrapedDeal, Deal, DealSelection, Opportunity
│   ├── scanner_agent.py          # RSS scraper + Groq JSON-mode deal selector
│   ├── frontier_agent.py         # Groq + ChromaDB RAG price estimator
│   ├── specialist_agent.py       # Groq fast specialist price estimator
│   ├── neural_network_agent.py   # PyTorch deep residual net price estimator
│   ├── ensemble_agent.py         # Weighted combination of the three models
│   ├── planning_agent.py         # Main pipeline orchestrator
│   ├── autonomous_planning_agent.py  # Alternative: autonomous tool-calling planner
│   ├── messaging_agent.py        # Groq notification writer + Pushover push
│   ├── preprocessor.py           # LiteLLM-based text normalizer (default: Groq)
│   ├── deep_neural_network.py    # PyTorch model architecture + inference
│   ├── items.py                  # Item dataclass for evaluation datasets
│   └── evaluator.py              # Tester class: MSE / R² / scatter charts
├── framework.py                  # DealAgentFramework: ChromaDB + memory + orchestration
├── app.py                        # Gradio web UI
├── log_utils.py                  # ANSI → HTML color converter for Gradio logs
├── setup_vectorstore.py          # One-time ChromaDB population from HuggingFace
├── requirements.txt
├── .env.example
└── README.md
```

---

## Using the Autonomous Agent (Optional)

To use the reasoning-based `AutonomousPlanningAgent` instead of the default
pipeline-based `PlanningAgent`, edit `framework.py`:

```python
# Change this import:
from agents.planning_agent import PlanningAgent
# To:
from agents.autonomous_planning_agent import AutonomousPlanningAgent as PlanningAgent
```

The autonomous agent uses Groq tool-calling to reason about which deals
are worth estimating and when to notify the user.

---

## Key Differences from the Original (week8)

| Feature | Original | BargainBuddy |
|---------|----------|--------------|
| Scanner LLM | OpenAI `gpt-5-mini` structured outputs | Groq `llama-3.3-70b-versatile` JSON mode |
| Frontier LLM | OpenAI `gpt-5.1` | Groq `llama-3.3-70b-versatile` |
| Autonomous planner | OpenAI `gpt-5.1` tool-calling | Groq `llama-3.3-70b-versatile` tool-calling |
| Specialist agent | Fine-tuned Llama on Modal | Groq `llama-3.1-8b-instant` with specialist prompt |
| Messaging | Claude Sonnet 4.5 via LiteLLM | Groq `llama-3.3-70b-versatile` via LiteLLM |
| Preprocessor | Ollama `llama3.2` (local) | Groq `llama-3.1-8b-instant` (cloud) |
| Infrastructure | OpenAI + Modal + Claude | Groq only |
