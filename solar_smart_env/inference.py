import os
import requests
import json
from openai import OpenAI
import time

from stable_baselines3 import PPO
from rl_agent import SolarGymEnv
from train_rl import train_agent, MODEL_PATH

_cached_model = None

def load_agent():
    global _cached_model
    if _cached_model is not None:
        return _cached_model
        
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print("Model not found. Training new agent...")
        train_agent(timesteps=50000)
    
    _cached_model = PPO.load(MODEL_PATH)
    return _cached_model

def get_action(obs_normalized):
    model = load_agent()
    if model:
        action, _states = model.predict(obs_normalized, deterministic=True)
        return int(action)
    return 1  # Default to 'distribute' if no model
# Variables fed by OpenEnv judging platform
API_BASE_URL = os.environ.get("API_BASE_URL", "https://huggingface.co/api/inference-proxy/together/v1") 
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN") 
API_KEY = os.environ.get("OPENAI_API_KEY", HF_TOKEN)

ENV_URL = "http://localhost:7860"

# Strict setup for inference compatibility
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

def run_task(task_name):
    # Reset Environment for this task
    try:
        res = requests.post(f"{ENV_URL}/reset", json={"task_name": task_name})
        obs = res.json()
    except Exception as e:
        print(f"Failed to connect to API: {e}")
        return

    print(f"[START] Task: {task_name}")
    
    done = False
    step_count = 0
    total_reward = 0.0
    info = {}

    while not done and step_count < 24:
        prompt = f"""
You are an AI managing a solar energy grid. Analyze the environment state and optimally route energy.
State: {json.dumps(obs)}

Actions:
0: Store energy (Solar -> Battery)
1: Distribute energy (Solar -> Houses)
2: Use Battery (Battery -> Houses)
3: Reduce load (Emergency survival)

Respond ONLY with the integer of the action (0, 1, 2, or 3).
"""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": "You are a programmatic controller. Output only a single digit."},{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0
            )
            # Find the first digit in the response
            reply = response.choices[0].message.content.strip()
            action = int([c for c in reply if c.isdigit()][0])
        except Exception:
            action = 1 # Fallback Distribute
            
        print(f"[STEP] Task {task_name} Step {step_count} Action {action}")
        
        # Post the action
        step_res = requests.post(f"{ENV_URL}/step", json={"action": action}).json()
        
        obs = step_res.get("observation", {})
        reward = step_res.get("reward", 0)
        done = step_res.get("done", True)
        info = step_res.get("info", {})
        
        total_reward += reward
        step_count += 1
        
    final_score = info.get("score", 0.0)
    print(f"[END] Task: {task_name} | Score: {final_score}")

if __name__ == "__main__":
    print("Waiting for OpenEnv server to accept connections...")
    # Wait for uvicorn to boot up
    time.sleep(3) 

    for task in ["easy", "medium", "hard"]:
        run_task(task)