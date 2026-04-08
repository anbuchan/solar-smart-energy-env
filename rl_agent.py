import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env import SolarEnergyEnv
from stable_baselines3 import PPO

MODEL_PATH = "ppo_solar_model"

class SolarGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = SolarEnergyEnv()
        
        # Exact 4 actions required: store(0), distribute(1), reduce_load(2), prioritize_critical(3)
        self.action_space = spaces.Discrete(4)
        
        # 13 Observation features based on env.state()
        # hour, solar, total_demand, hospital_demand, battery_charge, soc, health, price, rain + 4 houses
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(13,), dtype=np.float32)

    def _get_obs(self):
        s = self.env.state()
        return np.array([
            s["hour"] / 24.0,
            s["solar_generation"] / 1000.0,  
            s["total_demand"] / 1000.0,
            s["hospital_demand"] / 100.0,
            s["battery_charge"] / self.env.battery_capacity,
            s["battery_soc"],
            s["battery_health"],
            s["grid_price"] / 1.0,
            float(s["is_raining"]),
            s["per_house_demand"].get("house_1", 0) / 100.0,
            s["per_house_demand"].get("house_2", 0) / 100.0,
            s["per_house_demand"].get("house_3", 0) / 100.0,
            s["per_house_demand"].get("house_4", 0) / 100.0,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None, task_id="easy", lat="12.9716", lon="77.5946"):
        if seed is not None:
            np.random.seed(seed)
        _, info = self.env.reset(task_id=task_id, seed=seed, lat=lat, lon=lon)
        return self._get_obs(), info

    def step(self, action):
        _state, reward, done, info = self.env.step(int(action))
        truncated = False
        return self._get_obs(), reward, done, truncated, info

def get_trained_model():
    """Loads the PPO model or falls back gracefully without training loop."""
    try:
        return PPO.load(MODEL_PATH)
    except:
        return None
