import os
from huggingface_hub import InferenceClient

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

def get_rule_based_explanation(simulation_df):
    final_battery = simulation_df['Battery'].iloc[-1]
    avg_reward = simulation_df['Reward'].mean()
    
    # Simple rule-based logic to mimic AI explanation
    action_counts = simulation_df['Action'].value_counts()
    most_common = action_counts.idxmax() if not action_counts.empty else "Unknown"
    
    explanation = f"**Rule-based Analysis (API Token Missing):**\n\n"
    explanation += f"The agent prioritized '{most_common}' actions throughout the cycle. "
    
    if final_battery > 600:
        explanation += "The system finished with a highly charged battery, indicating the agent stored excess solar energy effectively. "
    elif final_battery < 200:
        explanation += "The system finished with a critically low battery, prioritizing current distribution over future storage. "
    else:
        explanation += "The system balanced distribution and storage adequately. "
        
    if avg_reward > 0.5:
        explanation += f"Overall efficiency was high (Reward: {avg_reward:.2f}), avoiding major blackouts."
    else:
        explanation += f"Overall efficiency was poor (Reward: {avg_reward:.2f}), with frequent blackouts or wasted energy."
        
    return explanation

def generate_xai_report(simulation_df, hf_token: str = None):
    if not hf_token or hf_token.strip() == "":
        return get_rule_based_explanation(simulation_df)
    
    try:
        client = InferenceClient(model=MODEL_ID, token=hf_token)
        
        stats_summary = simulation_df.describe().to_string()
        final_battery = simulation_df['Battery'].iloc[-1]
        avg_reward = simulation_df['Reward'].mean()
        
        messages = [
            {
                "role": "system", 
                "content": "You are an expert AI energy management auditor evaluating a Reinforcement Learning PPO agent managing a smart solar grid.\nYou must analyze the historical data, figure out why the AI took actions, and write a highly professional but accessible report.\nLimit your response to 2 concise paragraphs. Do not use generic filler words."
            },
            {
                "role": "user", 
                "content": f"Here is the 24-hour simulation statistical summary of the grid:\n{stats_summary}\n\nFinal Battery Storage: {final_battery:.2f} kWh\nAverage Effectiveness Reward Score: {avg_reward:.2f}\n\nWrite a professional Executive XAI Audit explaining the agent's performance, why it made certain decisions to balance solar generation vs housing demand, and declare the simulation a success or failure based on the final battery storage and average reward."
            }
        ]
        
        response = client.chat_completion(
            messages=messages,
            max_tokens=400,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ **Meta Llama 3 Inference Error:**\n```\n{e}\n```\n\nFallback:\n{get_rule_based_explanation(simulation_df)}"

