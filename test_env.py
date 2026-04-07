import pytest
from environment import TaxAwareRebalancerEnv
from models import Action, SellOrder

def test_hallucinated_lot_id():
    env = TaxAwareRebalancerEnv()
    env.reset()
    # LLM hallucinates a lot ID that doesn't exist. Added submit_portfolio=False
    bad_action = Action(
        sells=[SellOrder(ticker="VOO", quantity=10, lot_id="fake_id")],
        buys={},
        submit_portfolio=False
    )
    obs, reward, done, info = env.step(bad_action)

    # The environment should catch this and penalize, NOT crash
    assert "Invalid sell order" in str(info["errors"])

def test_wash_sale_enforcement():
    env = TaxAwareRebalancerEnv()
    obs = env.reset(task_level="hard")

    # 1. Force a loss harvest to trigger a wash sale restriction
    loss_lot = next(l for l in env.tax_lots if l.current_price < l.purchase_price)

    # Execute the sell. Added submit_portfolio=False
    env.step(Action(
        sells=[SellOrder(ticker=loss_lot.ticker, quantity=10, lot_id=loss_lot.lot_id)],
        buys={},
        submit_portfolio=False
    ))

    # 2. Try to immediately buy the restricted asset
    obs, reward, done, info = env.step(Action(
        buys={loss_lot.ticker: 10},
        sells=[],
        submit_portfolio=False
    ))

    assert loss_lot.ticker in obs.restricted_wash_sale_list
    assert "Wash sale violation" in str(info["errors"])