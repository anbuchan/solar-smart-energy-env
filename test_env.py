from env import SolarEnergyEnv
import json

def test():
    env = SolarEnergyEnv()
    
    print("--- Testing Reset ---")
    obs = env.reset(task_id="medium")
    print(f"Initial Observation: {json.dumps(obs, indent=2)}")
    
    print("\n--- Testing State ---")
    state = env.state()
    print(f"State: {json.dumps(state, indent=2)}")
    
    print("\n--- Testing Step (Action 1: Distribute) ---")
    next_obs, reward, done, info = env.step(1)
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {json.dumps(info, indent=2)}")
    
    print("\n--- Testing Step (Action 4: Export) ---")
    next_obs, reward, done, info = env.step(4)
    print(f"Reward: {reward}")
    print(f"Info: {json.dumps(info, indent=2)}")

if __name__ == "__main__":
    test()
