# OpenEnv: Tax-Aware Portfolio Rebalancer

## Real-World Motivation
Robo-advisors and wealth managers (e.g., Wealthfront, Betterment) manage billions of dollars using algorithmic tax-loss harvesting. When a portfolio drifts from its target allocation, selling assets to rebalance triggers capital gains taxes. This environment tests a frontier LLM's ability to execute complex, multi-step combinatorial optimization: hitting a target portfolio allocation while mathematically minimizing tax drag and adhering to IRS wash-sale regulations.

## Action & Observation Spaces (Strictly Typed via Pydantic)
* **Observation:** The agent sees `target_allocations`, available `cash`, a dynamically updated `restricted_wash_sale_list`, and a granular array of `TaxLots` (each containing purchase price, current price, and long-term/short-term tax status).
* **Action:** The agent outputs structured JSON dictating exact quantities to `buy()`, and an array of `sells` that target specific `lot_id` strings to execute tax-loss harvesting.

## The 3 Difficulty Tiers
1. **Easy (Basic Rebalance):** A tax-advantaged account (zero taxes). The agent must calculate $L_1$ distances and execute trades to reach the target allocation within a 1% margin of error, penalizing unnecessary transaction fees.
2. **Medium (Tax-Loss Harvesting):** Taxable accounts. The agent must selectively target specific tax lots operating at a loss to offset the necessary sale of lots operating at a gain, aiming for a net-zero capital gains bill.
3. **Hard (Wash Sale Avoidance):** Introduces temporal constraints. If the agent harvests a loss on an asset, it is dynamically added to a restricted list. The agent must comprehend this constraint, avoid buying the restricted asset, and identify proxy assets to satisfy the target portfolio allocation.

## Baseline Inference
The environment includes `inference.py`, a reproducible baseline utilizing `meta-llama/Meta-Llama-3-70B-Instruct` via the OpenAI client. The script forces strict JSON outputs corresponding to the Pydantic Action model.

### Setup Instructions
1. Install dependencies: `pip install openenv-core pydantic openai numpy`
2. Run validation: `openenv validate`
3. Execute baseline:
   ```bash
   export API_BASE_URL="[https://api-inference.huggingface.co/v1/](https://api-inference.huggingface.co/v1/)"
   export MODEL_NAME="meta-llama/Meta-Llama-3-70B-Instruct"
   export HF_TOKEN="your_token_here"
   python inference.py