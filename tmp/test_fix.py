import sys
import os
sys.path.append(os.getcwd())

from env import SolarEnergyEnv
import math

def test_env():
    print("Testing SolarEnergyEnv...")
    env = SolarEnergyEnv()
    
    # Test Reset
    print("Testing Reset...")
    res = env.reset(task_id="easy")
    assert isinstance(res, tuple), f"Expected tuple from reset, got {type(res)}"
    obs, info = res
    assert isinstance(obs, dict), f"Expected dict obs, got {type(obs)}"
    assert isinstance(info, dict), f"Expected dict info, got {type(info)}"
    assert "score" in info, "Score missing from info on reset"
    print(f"Initial Score: {info['score']}")
    assert 0.0 < info['score'] < 1.0, f"Score {info['score']} out of range (0, 1)"

    # Test Step
    print("Testing Step...")
    res = env.step(1) # Distribute
    assert isinstance(res, tuple), "Expected tuple from step"
    obs, reward, done, info = res
    assert "score" in info, "Score missing from info on step"
    print(f"Step Score: {info['score']}, Reward: {reward}")
    assert 0.0 < info['score'] < 1.0, f"Score {info['score']} out of range (0, 1)"
    
    # Test 24 steps
    print("Testing 24 steps...")
    for i in range(2, 25):
        obs, reward, done, info = env.step(1)
        score = info['score']
        assert 0.0 < score < 1.0, f"Score {score} out of range (0, 1) at step {i}"
        if done:
            print(f"Done at step {i}. Final Score: {score}")
            break
            
    print("Env tests passed!")

if __name__ == "__main__":
    test_env()
