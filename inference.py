import os
import json
from openai import OpenAI
from environment import TaxAwareRebalancerEnv

def run_inference():
    # 1. Mandatory Environment Variables Check
    api_key = os.environ.get("HF_TOKEN")
    base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")

    if not api_key:
        print("Warning: HF_TOKEN environment variable not set.")

    # 2. Mandatory OpenAI Client Usage
    client = OpenAI(
        base_url=base_url,
        api_key=api_key or "dummy_key"
    )

    env = TaxAwareRebalancerEnv()
    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        # MANDATORY LOG: [START]
        print(f"[START] task_id={task}")

        obs = env.reset(task_level=task)
        done = False
        step_num = 0

        while not done and step_num < 10:
            prompt = (
                f"You are a quantitative tax rebalancer.\n"
                f"Target Allocation: {env.target_alloc}\n"
                f"Current Cash: ${env.cash}\n"
                f"Tax Lots: {[lot.model_dump() for lot in env.tax_lots]}\n"
                f"Restricted Wash Sales: {env.restricted_list}\n"
                f"Output your action strictly as a JSON object with 'buys' (dict), 'sells' (list of dicts with ticker, quantity, lot_id), and 'submit_portfolio' (boolean)."
            )

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a JSON-only trading bot. Output valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    # Fallback to standard output if JSON mode isn't supported by the router
                    response_format={"type": "json_object"}
                )

                raw_action = response.choices[0].message.content
                action_dict = json.loads(raw_action)

                # MANDATORY LOG: [STEP]
                print(f"[STEP] step={step_num} action={json.dumps(action_dict)}")

                obs, reward, done, info = env.step(action_dict)

            except Exception as e:
                # If LLM hallucinates, log it and force an end to the episode
                fallback_action = {"buys": {}, "sells": [], "submit_portfolio": True}
                print(f"[STEP] step={step_num} action={json.dumps(fallback_action)} error={str(e)}")
                obs, reward, done, info = env.step(fallback_action)

            step_num += 1

        final_reward = max(0.0, float(reward))
        print(f"[END] task_id={task} reward={final_reward:.4f}")

if __name__ == "__main__":
    run_inference()
