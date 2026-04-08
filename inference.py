import os
import json
from openai import OpenAI
from environment import TaxAwareRebalancerEnv
from models import Action

def run_inference():
    # 1. Environment Variables Check
    api_key = os.environ.get("HF_TOKEN")
    base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

    if not api_key:
        print("Warning: HF_TOKEN environment variable not set. Using dummy key.")

    # 2. Strict OpenAI Client Usage (Required by Hackathon)
    client = OpenAI(
        base_url=base_url,
        api_key=api_key or "dummy_key"
    )

    env = TaxAwareRebalancerEnv()
    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        print(f"[START] task_id={task}")

        obs = env.reset(task_level=task)
        done = False
        step_num = 0
        reward = 0.0

        while not done and step_num < 10:
            prompt = (
                f"You are a quantitative tax rebalancer.\n"
                f"Target Allocation: {env.target_alloc}\n"
                f"Current Cash: ${env.cash}\n"
                f"Tax Lots: {[lot.model_dump() for lot in env.tax_lots]}\n"
                f"Restricted Wash Sales: {env.restricted_list}\n"
                f"Output your action strictly as a JSON object with 'reasoning' (string explaining your math), "
                f"'buys' (dict mapping ticker to quantity), 'sells' (list of dicts with ticker, quantity, lot_id), "
                f"and 'submit_portfolio' (boolean)."
            )

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a precise JSON-only trading algorithm. Output valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )

                raw_action = response.choices[0].message.content
                action_dict = json.loads(raw_action)

                print(f"[STEP] step={step_num} action={json.dumps(action_dict)}")
                obs, reward, done, info = env.step(Action(**action_dict))

            except Exception as e:
                # Safe Fallback to prevent complete crash on LLM hallucination
                fallback_action = {
                    "reasoning": "LLM failed to output valid JSON. Triggering automatic fallback.",
                    "buys": {},
                    "sells": [],
                    "submit_portfolio": True
                }
                print(f"[STEP] step={step_num} action={json.dumps(fallback_action)} error={str(e)}")
                obs, reward, done, info = env.step(Action(**fallback_action))

            step_num += 1

        final_reward = max(0.001, min(0.999, float(reward)))
        print(f"[END] task_id={task} reward={final_reward:.4f}")

if __name__ == "__main__":
    run_inference()