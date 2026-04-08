import sys
import os

# Add the subfolder to sys.path to simulate it being the CWD
subfolder_path = os.path.abspath("solar_smart_env")
sys.path.insert(0, subfolder_path)

try:
    from env import SolarEnergyEnv
    import math

    print("Verifying internal solar_smart_env/env.py...")
    env = SolarEnergyEnv()
    
    # Test reset
    state, info = env.reset(task_id="easy")
    print(f"Reset info: {info}")
    assert "score" in info, "Score missing from info on reset"
    assert 0.0 < info['score'] < 1.0, f"Initial score {info['score']} out of range (0, 1)"
    
    # Test steps
    for i in range(1, 25):
        state, reward, done, info = env.step(1) # Action 1: Distribute
        score = info['score']
        print(f"Step {i}: Score={score:.4f}")
        assert 0.0 < score < 1.0, f"Score {score} out of range (0, 1) at step {i}"
        if done:
            break
            
    print("\n✅ Verification SUCCESS: Internal environment is correctly graded and formatted.")

except Exception as e:
    print(f"\n❌ Verification FAILED: {e}")
    sys.exit(1)
