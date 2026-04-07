from pydantic import BaseModel, Field
from typing import Dict, List

<<<<<<< Updated upstream
# 1. Sub-Models (The Building Blocks)

=======
<<<<<<< HEAD
=======
# 1. Sub-Models (The Building Blocks)
>>>>>>> Stashed changes


>>>>>>> f968a3c741c44aa848c8fa18fbea7ad1a88bab00
class TaxLot(BaseModel):
    lot_id: str
    ticker: str
    quantity: int
    purchase_price: float
    current_price: float
    is_long_term: bool

class SellOrder(BaseModel):
<<<<<<< HEAD
    ticker: str
    quantity: int
    lot_id: str
=======
    ticker: str = Field(description="The asset ticker to sell.")
    quantity: int = Field(description="Number of shares to sell.")
    lot_id: str = Field(description="The specific tax lot ID to sell from. Crucial for tax-loss harvesting.")


# 2. The Observation Space (What the Agent Sees)


class Observation(BaseModel):
    step_number: int = Field(description="Current step in the environment.")
    current_cash: float = Field(description="Available cash balance to buy new assets.")
    target_allocation: Dict[str, float] = Field(description="Target portfolio percentages, e.g., {'VOO': 0.60, 'BND': 0.40}")
    current_allocation: Dict[str, float] = Field(description="Current portfolio percentages before action.")
    tax_lots: List[TaxLot] = Field(description="List of all currently held tax lots.")
    restricted_wash_sale_list: List[str] = Field(description="Tickers the agent is legally prohibited from buying this step due to recent loss-harvesting.")


# 3. The Action Space (What the Agent Does)

<<<<<<< Updated upstream
=======
>>>>>>> f968a3c741c44aa848c8fa18fbea7ad1a88bab00
>>>>>>> Stashed changes

class Action(BaseModel):
    reasoning: str = Field(
        description="Write out your step-by-step mathematical reasoning and tax calculations here BEFORE outputting the buys and sells."
    )
    buys: Dict[str, int] = Field(
        default_factory=dict,
        description="Dictionary mapping tickers to the quantity of shares to BUY."
    )
    sells: List[SellOrder] = Field(
        default_factory=list,
        description="List of specific sell orders, including exactly which tax lots to sell."
    )
    submit_portfolio: bool = Field(
        description="Set to True if the portfolio is perfectly rebalanced and the episode should end."
    )
<<<<<<< Updated upstream
=======
<<<<<<< HEAD

class Observation(BaseModel):
    cash: float
    tax_lots: List[TaxLot]
    target_alloc: Dict[str, float]
    restricted_wash_sale_list: List[str]
=======
>>>>>>> f968a3c741c44aa848c8fa18fbea7ad1a88bab00
>>>>>>> Stashed changes
