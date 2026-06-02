import json
import os
from typing import List

SUBSCRIBERS_FILE = "subscribers.json"


def load_subscribers() -> List[str]:
    if os.path.exists(SUBSCRIBERS_FILE):
        with open(SUBSCRIBERS_FILE) as f:
            return json.load(f)
    return []


def add_subscriber(user_key: str) -> int:
    """Add a user key if not already present. Returns total subscriber count."""
    subscribers = load_subscribers()
    if user_key not in subscribers:
        subscribers.append(user_key)
        with open(SUBSCRIBERS_FILE, "w") as f:
            json.dump(subscribers, f)
    return len(subscribers)
