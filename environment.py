import uuid
import numpy as np
from typing import Tuple, Dict, Any
from models import Observation, Action, TaxLot, SellOrder

class TaxAwareRebalancerEnv:
    def __init__(self):
        self.step_count = 0
        self.max_steps = 5
        self.task_level = "easy"
        self.cash = 0.0
        self.tax_lots = []
        self.target_alloc = {}
        self.restricted_list = []
        self.total_tax_paid = 0.0

    def reset(self, task_level: str = "easy") -> Observation:
        """
        Initializes the portfolio state.
        Easy/Medium levels provide a fixed baseline for reproducibility.
        Hard level provides stochastic state generation for robustness testing.
        """
        import random
        import uuid

        self.step_count = 0
        self.task_level = task_level
        self.total_tax_paid = 0.0
        self.restricted_list = []

        # Standard Target Allocation: 60% Equity (VOO), 40% Bonds (BND)
        self.target_alloc = {"VOO": 0.60, "BND": 0.40}

        # --- BRANCH: REPRODUCIBLE VS STOCHASTIC ---

        if task_level in ["easy", "medium"]:
            # Fixed starting state for standardized benchmarking
            self.cash = 10000.0
            self.tax_lots = [
                TaxLot(
                    lot_id="a1b2c3d4", ticker="VOO", quantity=100,
                    purchase_price=400.0, current_price=450.0, is_long_term=True
                ),
                TaxLot(
                    lot_id="e5f6g7h8", ticker="BND", quantity=200,
                    purchase_price=85.0, current_price=78.0, is_long_term=False
                ),
                TaxLot(
                    lot_id="i9j0k1l2", ticker="VOO", quantity=50,
                    purchase_price=480.0, current_price=450.0, is_long_term=False
                )
            ]
        else:
            # Stochastic generation for "Hard" mode to test AI generalization
            self.cash = round(random.uniform(5000.0, 15000.0), 2)
            num_lots = random.randint(4, 8)
            self.tax_lots = []

            for _ in range(num_lots):
                ticker = random.choice(["VOO", "BND"])
                quantity = random.randint(20, 150)

                # Dynamic pricing logic
                base_price = 450.0 if ticker == "VOO" else 72.0
                curr = base_price * random.uniform(0.95, 1.05)
                # Generate a mix of gains (low purchase price) and losses (high purchase price)
                purch = curr * random.uniform(0.75, 1.25)

                self.tax_lots.append(
                    TaxLot(
                        lot_id=str(uuid.uuid4())[:8],
                        ticker=ticker,
                        quantity=quantity,
                        purchase_price=round(purch, 2),
                        current_price=round(curr, 2),
                        is_long_term=random.choice([True, False])
                    )
                )

        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Executes trades, calculates capital gains, and returns the reward."""
        self.step_count += 1
        step_tax = 0.0
        transaction_fees = 0.0
        info = {"errors": []}

        # 1. Process Sells (Tax-Loss Harvesting Logic)
        for sell in action.sells:
            # Find the exact lot the LLM targeted
            lot = next((l for l in self.tax_lots if l.lot_id == sell.lot_id), None)
            if lot and lot.quantity >= sell.quantity:
                revenue = sell.quantity * lot.current_price
                cost_basis = sell.quantity * lot.purchase_price
                realized_gain = revenue - cost_basis

                # Apply tax rate (simplified: 15% long-term, 25% short-term)
                if self.task_level in ["medium", "hard"] and realized_gain > 0:
                    rate = 0.15 if lot.is_long_term else 0.25
                    step_tax += realized_gain * rate

                # Wash Sale constraints (Hard Mode)
                if self.task_level == "hard" and realized_gain < 0:
                    self.restricted_list.append(lot.ticker)

                self.cash += revenue
                lot.quantity -= sell.quantity
                transaction_fees += 5.0
            else:
                info["errors"].append(f"Invalid sell order for lot {sell.lot_id}")

        # Clean up empty lots
        self.tax_lots = [l for l in self.tax_lots if l.quantity > 0]

        # 2. Process Buys
        for ticker, qty in action.buys.items():
            if ticker in self.restricted_list:
                info["errors"].append(f"Wash sale violation: Cannot buy {ticker}")
                continue

            cost = qty * self._get_current_price(ticker)
            if cost <= self.cash:
                self.cash -= cost
                # Create new tax lot for the purchase
                self.tax_lots.append(
                    TaxLot(lot_id=str(uuid.uuid4())[:8], ticker=ticker, quantity=qty,
                           purchase_price=self._get_current_price(ticker), current_price=self._get_current_price(ticker), is_long_term=False)
                )
                transaction_fees += 5.0

        self.total_tax_paid += step_tax

        # 3. Calculate Reward (Minimize L1 Distance + Minimize Tax)
        current_alloc = self._calculate_allocation()

        # Vectorized L1 norm for allocation drift
        l1_distance = sum(abs(current_alloc.get(t, 0.0) - self.target_alloc.get(t, 0.0)) for t in self.target_alloc)

        # Reward shaping: Penalize drift, taxes, and unnecessary fees
        reward = -float(l1_distance) * 100.0 - (step_tax * 0.1) - (transaction_fees * 0.01)

        # Check termination
        done = action.submit_portfolio or self.step_count >= self.max_steps

        if done:
            # Final grading normalization (0.0 to 1.0) applied at the end of the episode
            info["final_l1_error"] = l1_distance
            info["total_tax"] = self.total_tax_paid

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> Observation:
        return Observation(
            step_number=self.step_count,
            current_cash=self.cash,
            target_allocation=self.target_alloc,
            current_allocation=self._calculate_allocation(),
            tax_lots=self.tax_lots,
            restricted_wash_sale_list=self.restricted_list
        )

    def _calculate_allocation(self) -> Dict[str, float]:
        portfolio_value = self.cash
        allocations = {}
        for lot in self.tax_lots:
            value = lot.quantity * lot.current_price
            portfolio_value += value
            allocations[lot.ticker] = allocations.get(lot.ticker, 0.0) + value

        if portfolio_value == 0: return {}
        return {k: v / portfolio_value for k, v in allocations.items()}

    def _get_current_price(self, ticker: str) -> float:
        prices = {"VOO": 450.0, "BND": 72.0, "SPY": 451.0} # SPY as proxy for VOO
        return prices.get(ticker, 100.0)

    def state(self):
        return self._get_observation().model_dump()