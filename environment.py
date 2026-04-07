import random
import uuid
from models import Observation, Action, TaxLot, SellOrder

class TaxAwareRebalancerEnv:
    def __init__(self):
        self.step_count = 0
        self.task_level = "easy"
        self.total_tax_paid = 0.0
        self.restricted_list = []
        self.cash = 0.0
        self.tax_lots = []
        self.target_alloc = {"VOO": 0.60, "BND": 0.40}

    def _get_observation(self):
        return Observation(
            cash=self.cash,
            tax_lots=self.tax_lots,
            target_alloc=self.target_alloc,
            restricted_wash_sale_list=self.restricted_list
        )

    def reset(self, task_level: str = "easy") -> Observation:
        self.step_count = 0
        self.task_level = task_level
        self.total_tax_paid = 0.0
        self.restricted_list = []
        self.target_alloc = {"VOO": 0.60, "BND": 0.40}

        # FIXED STATE for standardized benchmarking
        if task_level in ["easy", "medium"]:
            self.cash = 10000.0
            self.tax_lots = [
                TaxLot(lot_id="a1b2c3d4", ticker="VOO", quantity=100, purchase_price=400.0, current_price=450.0, is_long_term=True),
                TaxLot(lot_id="e5f6g7h8", ticker="BND", quantity=200, purchase_price=85.0, current_price=78.0, is_long_term=False),
                TaxLot(lot_id="i9j0k1l2", ticker="VOO", quantity=50, purchase_price=480.0, current_price=450.0, is_long_term=False)
            ]
        # STOCHASTIC STATE to prove AI generalization
        else:
            self.cash = round(random.uniform(5000.0, 15000.0), 2)
            num_lots = random.randint(4, 8)
            self.tax_lots = []
            for _ in range(num_lots):
                ticker = random.choice(["VOO", "BND"])
                quantity = random.randint(20, 150)
                base_price = 450.0 if ticker == "VOO" else 72.0
                curr = base_price * random.uniform(0.95, 1.05)
                purch = curr * random.uniform(0.75, 1.25)
                self.tax_lots.append(
                    TaxLot(lot_id=str(uuid.uuid4())[:8], ticker=ticker, quantity=quantity, purchase_price=round(purch, 2), current_price=round(curr, 2), is_long_term=random.choice([True, False]))
                )
        return self._get_observation()

    def step(self, action: Action):
        self.step_count += 1
        errors = []
        reward = 0.0
        done = False

        # 1. Process Sells & Taxes
        for sell in action.sells:
            lot = next((l for l in self.tax_lots if l.lot_id == sell.lot_id), None)
            if not lot or lot.quantity < sell.quantity:
                errors.append(f"Invalid sell order for lot {sell.lot_id}")
                continue

            proceeds = sell.quantity * lot.current_price
            cost_basis = sell.quantity * lot.purchase_price
            gain = proceeds - cost_basis

            if gain > 0:
                tax_rate = 0.15 if lot.is_long_term else 0.25
                self.total_tax_paid += gain * tax_rate
            elif gain < 0 and self.task_level == "hard":
                # Wash Sale Trap
                if lot.ticker not in self.restricted_list:
                    self.restricted_list.append(lot.ticker)

            self.cash += proceeds
            lot.quantity -= sell.quantity

        # 2. Process Buys
        for ticker, qty in action.buys.items():
            if self.task_level == "hard" and ticker in self.restricted_list:
                errors.append(f"Wash sale violation: {ticker} is restricted for 30 days.")
                continue

            price = 450.0 if ticker == "VOO" else 72.0
            cost = qty * price
            if cost > self.cash:
                errors.append(f"Insufficient cash to buy {qty} of {ticker}")
                continue

            self.cash -= cost
            self.tax_lots.append(TaxLot(lot_id=str(uuid.uuid4())[:8], ticker=ticker, quantity=qty, purchase_price=price, current_price=price, is_long_term=False))

        # Cleanup empty lots
        self.tax_lots = [l for l in self.tax_lots if l.quantity > 0]

        # 3. Terminate & Calculate Reward
        if action.submit_portfolio or self.step_count >= 10 or errors:
            done = True

            total_value = self.cash + sum(l.quantity * l.current_price for l in self.tax_lots)
            allocations = {"VOO": 0.0, "BND": 0.0}
            if total_value > 0:
                for l in self.tax_lots:
                    allocations[l.ticker] += (l.quantity * l.current_price) / total_value

            l1_error = sum(abs(self.target_alloc[t] - allocations.get(t, 0.0)) for t in self.target_alloc)

            # Reward Math: Max 1.0. Deduct for allocation drift, tax penalties, and rule breaks.
            base_reward = max(0.0, 1.0 - l1_error)
            tax_penalty = min(0.5, self.total_tax_paid / 5000.0)
            error_penalty = 0.5 if errors else 0.0

            reward = max(0.0, base_reward - tax_penalty - error_penalty)

        info = {"errors": errors, "total_tax_paid": round(self.total_tax_paid, 2)}
        return self._get_observation(), reward, done, info