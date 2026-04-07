import sys
import os
import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import TaxAwareRebalancerEnv

def main():
    print("Initializing Tax-Aware Rebalancer Environment Server...")
    env = TaxAwareRebalancerEnv()
    print("Environment loaded successfully. Ready for evaluation.")

if __name__ == "__main__":
    main()
