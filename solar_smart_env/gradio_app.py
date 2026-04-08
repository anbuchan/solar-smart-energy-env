import gradio as gr
import requests
import pandas as pd
import matplotlib.pyplot as plt
import time

# Assuming FastAPI runs locally or in same container
API_URL = "http://localhost:7860"

def format_action(action):
    mapping = {0: "Store 🔋", 1: "Distribute 🏠", 2: "Use Battery 🔋⚡", 3: "Reduce Load 🚨"}
    return mapping.get(action, "Unknown")

def run_simulation_ui():
    """Triggers the simulation via FastAPI and updates UI."""
    try:
        response = requests.get(f"{API_URL}/run?steps=24")
        data = response.json()["simulation_results"]
        
        # Convert to DataFrame for plotting
        df = pd.DataFrame([
            {
                "Step": r["step"],
                "Action": format_action(r["action"]),
                "Solar": r["observation"]["solar_generation"],
                "Battery": r["observation"]["battery_charge"],
                "Demand": r["observation"]["total_demand"],
                "Reward": r["reward"]
            } for r in data
        ])
        
        # Generate Plots
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        df.plot(x="Step", y=["Solar", "Demand"], ax=ax1, color=['gold', 'red'], marker='o')
        ax1.set_title("Solar Generation vs House Demand")
        ax1.grid(True)
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        df.plot(x="Step", y=["Battery"], ax=ax2, color='green', marker='s')
        ax2.set_title("Battery Storage Level 🔋")
        ax2.grid(True)
        
        # Calculate summary
        avg_reward = df["Reward"].mean()
        summary_text = f"### Simulation Complete 🚀\n- Average Efficiency (Reward): **{avg_reward:.2f}**\n- Final Battery Level: **{df['Battery'].iloc[-1]:.2f}**"
        
        return summary_text, fig1, fig2, df
        
    except Exception as e:
        return f"Error: {str(e)}", None, None, None

def get_history_ui():
    try:
        response = requests.get(f"{API_URL}/history")
        history = response.json()["history"]
        df = pd.DataFrame(history, columns=["ID", "Timestamp", "Step", "Action", "Solar", "Battery", "Demand", "Reward"])
        return df
    except:
        return "No history found."

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Smart Solar AI Command Center 🌞")
    gr.Markdown("Production-level Smart Grid Management powered by Reinforcement Learning (PPO).")
    
    with gr.Tab("Simulation Control"):
        with gr.Row():
            btn_run = gr.Button("Run AI Simulation 🚀", variant="primary")
            btn_reset = gr.Button("Reset Environment 🔄")
            
        status = gr.Markdown("Ready to simulate...")
        
        with gr.Row():
            plot_solar = gr.Plot(label="Solar vs Demand")
            plot_battery = gr.Plot(label="Battery Levels")
            
        logs = gr.DataFrame(label="Live Step Logs")
        
    with gr.Tab("Historical Analytics"):
        btn_hist = gr.Button("Refresh History 📊")
        history_table = gr.DataFrame()

    btn_run.click(run_simulation_ui, outputs=[status, plot_solar, plot_battery, logs])
    btn_hist.click(get_history_ui, outputs=[history_table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861) # Running on different port to not conflict if testing locally
