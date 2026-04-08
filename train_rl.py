from stable_baselines3 import PPO
from rl_agent import SolarGymEnv
import os

MODEL_PATH = "ppo_solar_model"

def train_agent(timesteps=10000):
    print("Starting RL Agent Training...")
    env = SolarGymEnv()
    
    # PPO agent initialization
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    
    # Train the agent
    print(f"Training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)
    
    # Save the model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    # Extensive training for a smart production agent
    train_agent(timesteps=50000)

