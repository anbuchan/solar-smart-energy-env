import gymnasium as gym
import numpy as np
import os
from gymnasium import spaces
from env import SolarEnergyEnv

MODEL_PATH = "ppo_solar_model"

class SolarGymEnv(gym.Env):
    def __init__(self, num_houses=4):
        super(SolarGymEnv, self).__init__()
        self.num_houses = num_houses
        self.env = SolarEnergyEnv(num_houses=num_houses)
        
        # Action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)
        
        # Observation space: hour, solar, battery_charge, battery_health, battery_efficiency
        # plus individual house demands (num_houses)
        obs_length = 5 + self.num_houses
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_length,), dtype=np.float32)

    def _normalize_obs(self, obs_dict):
        # Extract base scalars
        base_features = [
            obs_dict["hour"] / 24.0,
            min(obs_dict["solar_generation"] / 1000.0, 1.0),
            min(obs_dict["battery_charge"] / self.env.battery_capacity, 1.0),
            min(obs_dict["battery_health"], 1.0),
            min(obs_dict["battery_efficiency"], 1.0)
        ]
        
        # Extract per-house demands mapping (assuming "house_1", "house_2", etc.)
        house_features = []
        for i in range(1, self.num_houses + 1):
            key = f"house_{i}"
            demand = obs_dict["per_house_demand"].get(key, 0.0)
            house_features.append(min(demand / 100.0, 1.0))
            
        return np.array(base_features + house_features, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_dict = self.env.reset(seed=seed)
        return self._normalize_obs(obs_dict), {}

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        truncated = False
        return self._normalize_obs(obs_dict), reward, done, truncated, info

def get_trained_model():
    from stable_baselines3 import PPO
    import os
    
    if os.path.exists(f"{MODEL_PATH}.zip"):
        return PPO.load(MODEL_PATH)
    
    print(f"Model {MODEL_PATH}.zip not found. Auto-training lightweight fallback model...")
    # Auto-train lightweight fallback
    env = SolarGymEnv()
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=3000)
    model.save(MODEL_PATH)
    return model
