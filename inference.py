import os
import json
import requests
from openai import OpenAI

# 1. Required Read from Environment Variables per Meta OpenEnv Hackathon Spec
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required for winning-level submission.")

# 2. Initialize OpenAI Client strictly using the specified SDK
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def emit_log(line_type, **kwargs):
    """Ensures exact single-line log formatting for the automated judge."""
    if line_type == "START":
        print(f"[START] task={kwargs['task']} env={kwargs['env']} model={kwargs['model']}")
    elif line_type == "STEP":
        # Format reward to 2 decimal places, done and success to lowercase booleans
        reward_str = f"{float(kwargs['reward']):.2f}"
        done_str = str(kwargs['done']).lower()
        error_str = str(kwargs.get('error', 'null')).replace('\n', ' ') if kwargs.get('error') else "null"
        print(f"[STEP] step={kwargs['step']} action={kwargs['action']} reward={reward_str} done={done_str} error={error_str}")
    elif line_type == "END":
        success_str = str(kwargs['success']).lower()
        rewards_list_str = ",".join([f"{float(r):.2f}" for r in kwargs['rewards']])
        print(f"[END] success={success_str} steps={kwargs['steps']} rewards={rewards_list_str}")

def run_task(task_name):
    # Connect to the local Solar Smart Grid environment
    # Note: Using direct import here to simplify container execution
    from env import SolarEnergyEnv
    env = SolarEnergyEnv()
    
    try:
        obs = env.reset(task_id=task_name)
    except Exception as e:
        # Standard error handling
        return

    # [START] emit
    emit_log("START", task=task_name, env="solar-smart-grid", model=MODEL_NAME)
    
    done = False
    step_count = 1
    rewards = []
    success = False
    
    # Action Mapping for log consistency
    action_map = {
        0: "store_energy",
        1: "distribute_energy",
        2: "reduce_load",
        3: "prioritize_critical"
    }

    try:
        while not done:
            prompt = f"""You are an AI grid controller. Optimize the solar grid.
State: {json.dumps(obs)}
Actions: 0:store_energy, 1:distribute_energy, 2:reduce_load, 3:prioritize_critical.
Output ONLY the integer action."""
            
            last_action_error = None
            action_int = 1 # Default fallback
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0.0
                )
                reply = response.choices[0].message.content.strip()
                digits = [c for c in reply if c.isdigit()]
                if digits:
                    action_int = int(digits[0])
                if action_int not in [0, 1, 2, 3]:
                    action_int = 1
            except Exception as e:
                last_action_error = str(e)
                action_int = 1

            # Environment Step
            try:
                obs, reward, done, info = env.step(action_int)
            except Exception as e:
                last_action_error = str(e)
                reward = -1.0
                done = True
                
            rewards.append(reward)
            emit_log("STEP", step=step_count, action=action_map.get(action_int, "distribute_energy"), 
                     reward=reward, done=done, error=last_action_error)
            
            step_count += 1
            if done:
                success = info.get("score", 0.0) > 0.5
                break
    except Exception as e:
        print(f"Error during task execution: {e}")
    finally:
        # Mandatory OpenEnv Lifecycle: Close env THEN emit END
        env.close()
        emit_log("END", success=success, steps=len(rewards), rewards=rewards)

if __name__ == "__main__":
    # The platform judge will call inference.py for specific tasks
    for task in ["easy", "medium", "hard"]:
        run_task(task)