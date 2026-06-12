"""Synthetic transaction generator and live-feed simulator.

Generates realistic card transactions for synthetic Australian customers and
injects three classic fraud patterns: stolen-card spree, account takeover and
card testing. The same machinery produces both the model's training data and
the live feed in the demo app, so feature distributions match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# category: (risk weight, typical amount mu, sigma)  — lognormal params, AUD
CATEGORIES = {
    "grocery": (0.05, 4.0, 0.5),
    "fuel": (0.05, 4.2, 0.3),
    "dining": (0.15, 3.8, 0.6),
    "pharmacy": (0.05, 3.4, 0.5),
    "retail": (0.25, 4.3, 0.7),
    "streaming": (0.10, 2.6, 0.3),
    "travel": (0.45, 5.8, 0.6),
    "online_marketplace": (0.50, 4.4, 0.8),
    "electronics": (0.60, 6.0, 0.6),
    "jewellery": (0.70, 6.3, 0.7),
    "gambling": (0.80, 4.8, 0.9),
    "gift_cards": (0.90, 4.6, 0.7),
}

CITIES = {
    "Melbourne": (-37.81, 144.96),
    "Sydney": (-33.87, 151.21),
    "Brisbane": (-27.47, 153.03),
    "Perth": (-31.95, 115.86),
    "Adelaide": (-34.93, 138.60),
    "Hobart": (-42.88, 147.33),
    "Canberra": (-35.28, 149.13),
    "Darwin": (-12.46, 130.84),
}

EVERYDAY = ["grocery", "fuel", "dining", "pharmacy", "retail", "streaming"]
SPREE_TARGETS = ["electronics", "jewellery", "gift_cards", "retail"]
ONLINE_ONLY = ["online_marketplace", "gift_cards", "gambling", "streaming"]

FEATURES = [
    "amount",
    "log_amount",
    "hour",
    "is_night",
    "merchant_risk",
    "is_online",
    "dist_from_home_km",
    "mins_since_prev",
    "txns_last_hour",
    "amount_ratio",
    "is_new_merchant",
]

SCENARIOS = {
    "stolen_card": "Stolen card spree",
    "account_takeover": "Account takeover",
    "card_testing": "Card testing",
}


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371 * math.asin(math.sqrt(h))


def make_customers(n: int, rng: np.random.Generator) -> list[dict]:
    customers = []
    cities = list(CITIES)
    for i in range(n):
        city = cities[rng.integers(len(cities))]
        customers.append(
            {
                "customer_id": f"C{i:04d}",
                "home_city": city,
                "avg_amount": float(np.exp(rng.normal(4.0, 0.4))),  # ~AUD 55 typical
                "night_owl": float(rng.uniform(0, 0.25)),
                "online_rate": float(rng.uniform(0.1, 0.4)),
            }
        )
    return customers


@dataclass
class _History:
    """Rolling per-customer history used for feature engineering."""

    times: list[datetime] = field(default_factory=list)
    merchants: set = field(default_factory=set)


class FeedSimulator:
    """Produces scored-ready transactions, one batch or fraud episode at a time."""

    def __init__(self, n_customers: int = 150, seed: int = 7,
                 start: datetime | None = None):
        self.rng = np.random.default_rng(seed)
        self.customers = make_customers(n_customers, self.rng)
        self.now = start or datetime(2026, 6, 1, 9, 0)
        self.histories: dict[str, _History] = {
            c["customer_id"]: _History() for c in self.customers
        }
        self._txn_counter = 0

    # ---------- internals ----------

    def _merchant_name(self, category: str) -> str:
        return f"{category}_{self.rng.integers(1, 40):02d}"

    def _featurize(self, cust: dict, ts: datetime, amount: float, category: str,
                   merchant: str, city: str | None, is_online: bool,
                   is_fraud: bool, scenario: str) -> dict:
        hist = self.histories[cust["customer_id"]]
        home = CITIES[cust["home_city"]]
        if is_online:
            dist = 0.0
        else:
            base = CITIES[city] if city in CITIES else home
            jitter = (float(self.rng.normal(0, 0.05)), float(self.rng.normal(0, 0.05)))
            dist = haversine_km(home, (base[0] + jitter[0], base[1] + jitter[1]))

        prev = hist.times[-1] if hist.times else None
        mins_since_prev = min((ts - prev).total_seconds() / 60, 1440.0) if prev else 1440.0
        txns_last_hour = sum(1 for t in hist.times if 0 <= (ts - t).total_seconds() <= 3600)
        is_new_merchant = int(merchant not in hist.merchants)

        self._txn_counter += 1
        row = {
            "txn_id": f"T{self._txn_counter:06d}",
            "timestamp": ts,
            "customer_id": cust["customer_id"],
            "merchant": merchant,
            "category": category,
            "city": "online" if is_online else (city or cust["home_city"]),
            "amount": round(amount, 2),
            "log_amount": math.log1p(amount),
            "hour": ts.hour,
            "is_night": int(ts.hour >= 22 or ts.hour < 6),
            "merchant_risk": CATEGORIES[category][0],
            "is_online": int(is_online),
            "dist_from_home_km": round(dist, 1),
            "mins_since_prev": round(mins_since_prev, 1),
            "txns_last_hour": txns_last_hour,
            "amount_ratio": round(amount / cust["avg_amount"], 2),
            "is_new_merchant": is_new_merchant,
            "is_fraud": int(is_fraud),
            "scenario": scenario,
        }
        hist.times.append(ts)
        if len(hist.times) > 200:
            del hist.times[:100]
        hist.merchants.add(merchant)
        return row

    def _legit_txn(self, cust: dict, ts: datetime) -> dict:
        category = EVERYDAY[self.rng.integers(len(EVERYDAY))]
        r = self.rng.random()
        # hard negatives: legit behaviour that superficially resembles fraud
        if r < 0.06:
            category = "travel"
        elif r < 0.10:  # big-ticket purchase (new TV, ring, holiday booking)
            category = ["electronics", "jewellery"][self.rng.integers(2)]
        elif r < 0.16:  # routine online shopping
            category = "online_marketplace"
        elif r < 0.18:  # recreational gambling
            category = "gambling"
        _, mu, sigma = CATEGORIES[category]
        scale = cust["avg_amount"] / math.exp(4.0)
        amount = float(np.exp(self.rng.normal(mu, sigma))) * scale
        is_online = self.rng.random() < cust["online_rate"] or category == "streaming"
        city = cust["home_city"]
        if not is_online and self.rng.random() < 0.03:  # legit interstate travel
            city = list(CITIES)[self.rng.integers(len(CITIES))]
        merchant = self._merchant_name(category)
        # regular customers mostly revisit known merchants
        hist = self.histories[cust["customer_id"]]
        if hist.merchants and self.rng.random() < 0.7:
            merchant = list(hist.merchants)[self.rng.integers(len(hist.merchants))]
        return self._featurize(cust, ts, amount, category, merchant, city,
                               is_online, False, "")

    def _legit_hour(self, cust: dict) -> int:
        if self.rng.random() < cust["night_owl"]:
            return int(self.rng.integers(20, 24))
        return int(np.clip(self.rng.normal(14, 4), 7, 21))

    # ---------- public API ----------

    def stream(self, n: int = 25, fraud_rate: float = 0.01) -> pd.DataFrame:
        """Advance the clock ~2h and emit n transactions across the book."""
        rows = []
        for _ in range(n):
            self.now += timedelta(minutes=float(self.rng.uniform(2, 9)))
            cust = self.customers[self.rng.integers(len(self.customers))]
            ts = self.now.replace(hour=self._legit_hour(cust),
                                  minute=int(self.rng.integers(60)))
            r = self.rng.random()
            if r < fraud_rate:
                rows.extend(self._episode(cust, ts))
            elif r < fraud_rate + 0.04:
                rows.extend(self._shopping_trip(cust, ts))
            else:
                rows.append(self._legit_txn(cust, ts))
        return pd.DataFrame(rows)

    def _shopping_trip(self, cust: dict, ts: datetime) -> list[dict]:
        """Legit burst: several purchases within an hour at the local mall."""
        rows = []
        for _ in range(int(self.rng.integers(3, 7))):
            ts += timedelta(minutes=float(self.rng.uniform(5, 20)))
            rows.append(self._legit_txn(cust, ts))
        return rows

    def inject(self, scenario: str) -> pd.DataFrame:
        """Inject a named fraud episode for a random customer at the current time."""
        cust = self.customers[self.rng.integers(len(self.customers))]
        self.now += timedelta(minutes=5)
        return pd.DataFrame(self._episode(cust, self.now, scenario))

    def _episode(self, cust: dict, ts: datetime, scenario: str | None = None) -> list[dict]:
        scenario = scenario or list(SCENARIOS)[self.rng.integers(len(SCENARIOS))]
        rows = []
        if scenario == "stolen_card":
            # burst of card-present purchases far from home, escalating amounts
            city = list(CITIES)[self.rng.integers(len(CITIES))]
            amount = float(self.rng.uniform(80, 220))
            for i in range(int(self.rng.integers(6, 11))):
                ts += timedelta(minutes=float(self.rng.uniform(3, 14)))
                category = SPREE_TARGETS[self.rng.integers(len(SPREE_TARGETS))]
                rows.append(self._featurize(cust, ts, amount,
                                            category, self._merchant_name(category),
                                            city, False, True, scenario))
                amount *= float(self.rng.uniform(1.2, 1.8))
        elif scenario == "account_takeover":
            # late-night online purchases at never-seen merchants
            ts = ts.replace(hour=int(self.rng.integers(0, 5)))
            for _ in range(int(self.rng.integers(4, 8))):
                ts += timedelta(minutes=float(self.rng.uniform(4, 20)))
                category = ONLINE_ONLY[self.rng.integers(len(ONLINE_ONLY))]
                amount = float(self.rng.uniform(150, 1200))
                rows.append(self._featurize(cust, ts, amount,
                                            category, self._merchant_name(category),
                                            None, True, True, scenario))
        elif scenario == "card_testing":
            # rapid micro-charges to validate the card, then big hits
            for _ in range(int(self.rng.integers(8, 15))):
                ts += timedelta(seconds=float(self.rng.uniform(20, 90)))
                amount = float(self.rng.uniform(0.5, 3.0))
                rows.append(self._featurize(cust, ts, amount, "online_marketplace",
                                            self._merchant_name("online_marketplace"),
                                            None, True, True, scenario))
            for _ in range(int(self.rng.integers(1, 4))):
                ts += timedelta(minutes=float(self.rng.uniform(2, 8)))
                amount = float(self.rng.uniform(300, 1500))
                category = ["electronics", "gift_cards"][self.rng.integers(2)]
                rows.append(self._featurize(cust, ts, amount, category,
                                            self._merchant_name(category),
                                            None, True, True, scenario))
        else:
            raise ValueError(f"unknown scenario: {scenario}")
        return rows


def build_training_data(n_customers: int = 150, n_batches: int = 60,
                        seed: int = 42) -> pd.DataFrame:
    """Simulate weeks of activity with ~4% fraud episodes for model training."""
    sim = FeedSimulator(n_customers=n_customers, seed=seed,
                        start=datetime(2026, 1, 5, 9, 0))
    frames = [sim.stream(n=120, fraud_rate=0.012) for _ in range(n_batches)]
    return pd.concat(frames, ignore_index=True)
