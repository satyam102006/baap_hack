from pydantic import BaseModel, Field
from typing import Dict, List

class TaxLot(BaseModel):
    lot_id: str
    ticker: str
    quantity: int
    purchase_price: float
    current_price: float
    is_long_term: bool

class SellOrder(BaseModel):
    ticker: str
    quantity: int
    lot_id: str

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

class Observation(BaseModel):
    cash: float
    tax_lots: List[TaxLot]
    target_alloc: Dict[str, float]
    restricted_wash_sale_list: List[str]