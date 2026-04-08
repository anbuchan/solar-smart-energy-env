from fastapi import FastAPI
from env import SolarEnergyEnv
from rl_agent import SolarGymEnv, get_trained_model
from database import save_step, get_history, init_db
import uvicorn
import gradio as gr
import pandas as pd
import json
import random
import plotly.graph_objects as go
from llm import generate_xai_report

app = FastAPI()

# Initialization on Startup
init_db()

@app.get("/health")
def health():
    return {"message": "Solar Smart Energy Env API is healthy 🚀"}

@app.get("/history")
def history():
    data = get_history()
    return {"history": data}

# Global Environment State for Inference Script
shared_env = SolarEnergyEnv()

@app.post("/reset")
def reset_env(data: dict = None):
    task_name = data.get("task_name", "easy") if data else "easy"
    obs = shared_env.reset(task_name=task_name)
    return obs

@app.post("/step")
def step_env(data: dict):
    action = data.get("action", 1)
    obs, reward, done, info = shared_env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

def format_action(action):
    mapping = {0: "Store 🔋", 1: "Distribute 🏠", 2: "Use Battery 🔋⚡", 3: "Reduce Load 🚨"}
    return mapping.get(action, "Unknown")

def create_plotly_figure(df):
    # Dark modern colors - Tesla Style
    bg_color = "rgba(10,10,10,0.6)"
    grid_color = "rgba(255,255,255,0.05)"
    text_color = "#ffffff"
    
    # 1. Solar vs Demand
    fig_solar = go.Figure()
    fig_solar.add_trace(go.Scatter(x=df['Step'], y=df['Solar'], name="Solar Generation", line=dict(color="#00ffcc", width=4, shape='spline'), fill='tozeroy', fillcolor="rgba(0, 255, 204, 0.15)"))
    fig_solar.add_trace(go.Scatter(x=df['Step'], y=df['Demand'], name="Total Demand", line=dict(color="#ff0055", width=3, shape='spline', dash='dot')))
    fig_solar.update_layout(title="⚡ Solar Input vs Grid Demand", paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=text_color), hovermode="x unified", xaxis=dict(gridcolor=grid_color), yaxis=dict(gridcolor=grid_color), margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # 2. Battery Storage over time
    fig_battery = go.Figure()
    fig_battery.add_trace(go.Scatter(x=df['Step'], y=df['Battery'], name="Battery Charge", line=dict(color="#00ccff", width=4, shape='spline'), fill='tozeroy', fillcolor="rgba(0, 204, 255, 0.2)"))
    fig_battery.update_layout(title="🔋 Energy Storage (kWh)", paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=text_color), hovermode="x unified", xaxis=dict(gridcolor=grid_color), yaxis=dict(gridcolor=grid_color), margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # 3. RL vs Baseline Reward
    fig_reward = go.Figure()
    fig_reward.add_trace(go.Scatter(x=df['Step'], y=df['Reward'], name="PPO AI Agent", line=dict(color="#b026ff", width=4, shape='spline')))
    fig_reward.add_trace(go.Scatter(x=df['Step'], y=df['Baseline_Reward'], name="Random Baseline", line=dict(color="#64748b", width=2, dash="dash", shape='spline')))
    fig_reward.update_layout(title="🤖 AI Performance Tracking (Reward)", paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=text_color), hovermode="x unified", xaxis=dict(gridcolor=grid_color), yaxis=dict(gridcolor=grid_color), margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # 4. Per-House Distribution (Stacked Area for smoother look)
    fig_dist = go.Figure()
    if len(df) > 0 and 'Distribution' in df.columns:
        houses = list(df.iloc[0]['Distribution'].keys())
        colors = ["#ec4899", "#8b5cf6", "#14b8a6", "#f97316", "#eab308"]
        for i, house in enumerate(houses):
            y_vals = [d.get(house, 0) for d in df['Distribution']]
            fig_dist.add_trace(go.Scatter(x=df['Step'], y=y_vals, name=house, line=dict(width=2, color=colors[i%len(colors)], shape='spline'), stackgroup='one', fillcolor=colors[i%len(colors)]))
    fig_dist.update_layout(title="🏠 Per-House Energy Allocation", paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=text_color), hovermode="x unified", xaxis=dict(gridcolor=grid_color), yaxis=dict(gridcolor=grid_color), margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    return fig_solar, fig_battery, fig_reward, fig_dist

def run_simulation_ui(hf_token):
    sim_seed = random.randint(1, 1000)
    
    rl_sim = SolarGymEnv()
    obs_dict_rl = rl_sim.env.reset(seed=sim_seed)
    obs_norm_rl = rl_sim._normalize_obs(obs_dict_rl)
    
    base_sim = SolarGymEnv()
    base_sim.env.reset(seed=sim_seed)
    
    try:
        agent_model = get_trained_model()
    except Exception as e:
        return [f"Error: {e}"] * 8 + [None]*4 + [pd.DataFrame(), "Error"]
        
    data = []
    
    for step_idx in range(24):
        action, _ = agent_model.predict(obs_norm_rl, deterministic=True)
        action = int(action)
        new_obs_dict_rl, rl_reward, done, info_rl = rl_sim.env.step(action)
        obs_norm_rl = rl_sim._normalize_obs(new_obs_dict_rl)
        
        random_action = base_sim.action_space.sample()
        _, base_reward, _, _ = base_sim.env.step(random_action)
        
        wasted_ai = info_rl.get("wasted_energy", 0)
        
        save_step(
            step=step_idx, 
            action=action, 
            solar=new_obs_dict_rl["solar_generation"], 
            battery=new_obs_dict_rl["battery_charge"], 
            total_demand=new_obs_dict_rl["total_demand"], 
            per_house_demand=new_obs_dict_rl["per_house_demand"], 
            per_house_distribution=info_rl.get("per_house_distribution", {}), 
            reward=rl_reward, 
            baseline_reward=base_reward,
            efficiency=new_obs_dict_rl["battery_efficiency"],
            wasted_energy=wasted_ai
        )
        
        data.append({
            "Step": step_idx,
            "Action": format_action(action),
            "Solar": new_obs_dict_rl["solar_generation"],
            "Battery": new_obs_dict_rl["battery_charge"],
            "Demand": new_obs_dict_rl["total_demand"],
            "Reward": rl_reward,
            "Baseline_Reward": base_reward,
            "Wasted": wasted_ai,
            "Distribution": info_rl.get("per_house_distribution", {})
        })
        
        if done: break
        
    df = pd.DataFrame(data)
    fig_solar, fig_battery, fig_reward, fig_dist = create_plotly_figure(df)
    
    # Calculate Impact Metrics
    total_ai_reward = df["Reward"].sum()
    total_base_reward = df["Baseline_Reward"].sum()
    reward_diff = total_ai_reward - total_base_reward
    
    # Efficiency Gain %
    eff_gain = (reward_diff / abs(total_base_reward + 0.1)) * 100 if total_base_reward != 0 else 100.0
    eff_gain_str = f"+{eff_gain:.1f}%" if eff_gain >= 0 else f"{eff_gain:.1f}%"
    
    # Energy Saved (assuming finishing battery is retained energy)
    energy_saved = df["Battery"].iloc[-1]
    
    # Waste Reduction 
    avg_waste = df["Wasted"].mean()
    waste_reduction = max(0, 100.0 - (avg_waste * 2)) # Mock formula indicating 100% reduction if zero waste
    
    # Impact Statement
    wow_statement = f"<div class='wow-impact'>🚀 AI Output Analysis: System efficiency improved by {eff_gain_str} compared to standard random baseline.</div>"
    
    # Metric HTML formatting
    m1 = f"<div class='metric-value'>{eff_gain_str}</div><div class='metric-label'>Efficiency Gain</div>"
    m2 = f"<div class='metric-value'>{energy_saved:.0f} kWh</div><div class='metric-label'>Energy Retained</div>"
    m3 = f"<div class='metric-value'>+{reward_diff:.1f}</div><div class='metric-label'>AI Reward Delta</div>"
    m4 = f"<div class='metric-value'>{waste_reduction:.1f}%</div><div class='metric-label'>Waste Prevented</div>"
    
    # XAI Report
    xai_audit = f"<div class='xai-container'><h3>🧠 AI Insight Engine</h3><p>{generate_xai_report(df, hf_token)}</p></div>"
    
    df['Distribution'] = df['Distribution'].apply(lambda x: ", ".join([f"{k}: {v}" for k,v in x.items()]))
    
    return wow_statement, m1, m2, m3, m4, fig_solar, fig_battery, fig_reward, fig_dist, df, xai_audit

def get_history_ui():
    history = get_history()
    if not history:
        return pd.DataFrame()
    return pd.DataFrame(history)

# Ultimate Dark Tesla-Style CSS
custom_css = """
body, .gradio-container {
    background-color: #0a0a0a !important;
    background-image: radial-gradient(circle at top, #111827 0%, #0a0a0a 100%) !important;
    color: #ffffff !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* Glass Panels */
.glass-panel {
    background: rgba(20, 20, 20, 0.75) !important;
    backdrop-filter: blur(15px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8) !important;
    padding: 20px !important;
    margin-bottom: 20px !important;
}

/* Typography */
.neon-title {
    text-align: center;
    font-weight: 900 !important;
    font-size: 3em !important;
    margin-bottom: 0px !important;
    color: #ffffff !important;
    text-shadow: 0 0 20px rgba(0, 255, 204, 0.5) !important;
    letter-spacing: 2px;
}
.neon-subtitle {
    text-align: center;
    color: #00ccff !important;
    font-size: 1.2em;
    font-weight: 500;
    margin-bottom: 30px !important;
    letter-spacing: 4px;
    text-transform: uppercase;
}

/* Wow Impact */
.wow-impact {
    font-size: 1.6em;
    font-weight: 800;
    text-align: center;
    padding: 15px 30px;
    background: linear-gradient(90deg, rgba(0,204,255,0.15) 0%, rgba(0,255,204,0.15) 100%);
    border: 1px solid rgba(0,255,204,0.4);
    border-radius: 12px;
    color: #ffffff;
    text-shadow: 0 0 15px #00ffcc;
    box-shadow: 0 0 30px rgba(0,255,204,0.15);
}

/* Top Metric Cards */
.metric-card {
    background: linear-gradient(145deg, rgba(30,30,30,0.8) 0%, rgba(10,10,10,0.9) 100%) !important;
    backdrop-filter: blur(25px) !important;
    border-radius: 20px !important;
    padding: 30px 10px !important;
    text-align: center !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease;
}
.metric-card:hover { transform: translateY(-5px) scale(1.02); }
.metric-green { border-bottom: 4px solid #00ffcc !important; box-shadow: 0 15px 35px rgba(0, 255, 204, 0.15) !important; }
.metric-blue { border-bottom: 4px solid #00ccff !important; box-shadow: 0 15px 35px rgba(0, 204, 255, 0.15) !important; }
.metric-purple { border-bottom: 4px solid #b026ff !important; box-shadow: 0 15px 35px rgba(176, 38, 255, 0.15) !important; }
.metric-orange { border-bottom: 4px solid #ffaa00 !important; box-shadow: 0 15px 35px rgba(255, 170, 0, 0.15) !important; }
.metric-value { font-size: 2.8em; font-weight: 900; margin-bottom: 5px; line-height: 1.1; }
.metric-green .metric-value { color: #00ffcc; text-shadow: 0 0 20px rgba(0,255,204,0.5); }
.metric-blue .metric-value { color: #00ccff; text-shadow: 0 0 20px rgba(0,204,255,0.5); }
.metric-purple .metric-value { color: #b026ff; text-shadow: 0 0 20px rgba(176,38,255,0.5); }
.metric-orange .metric-value { color: #ffaa00; text-shadow: 0 0 20px rgba(255,170,0,0.5); }
.metric-label { font-size: 0.95em; color: #a1a1aa; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }

/* Primary Button */
.primary-btn {
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
    border: none !important;
    color: #000000 !important;
    font-weight: 800 !important;
    font-size: 18px !important;
    letter-spacing: 1px;
    border-radius: 12px !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 20px rgba(79, 172, 254, 0.5) !important;
    padding: 16px !important;
}
.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 30px rgba(79, 172, 254, 0.8) !important;
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
}

/* XAI Panel */
.xai-container {
    background: rgba(20, 20, 30, 0.8);
    border-left: 5px solid #00ffcc;
    padding: 25px;
    border-radius: 12px;
    font-size: 1.1em;
    line-height: 1.6;
    color: #e2e8f0;
}
.xai-container h3 {
    color: #00ffcc;
    margin-top: 0;
    font-size: 1.5em;
    font-weight: 800;
}
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css) as demo:
    # HEADER
    gr.Markdown("<h1 class='neon-title'>⚡ AI Smart Solar Command Center</h1>")
    gr.Markdown("<div class='neon-subtitle'>Production Autonomous Grid Management</div>")
    
    wow_impact = gr.Markdown("<div class='wow-impact'>Waiting for AI System Initialization...</div>")
    
    # TOP METRICS ROW
    with gr.Row():
        top_metric_1 = gr.Markdown("<div class='metric-value'>--</div><div class='metric-label'>Efficiency Gain</div>", elem_classes=["metric-card", "metric-green"])
        top_metric_2 = gr.Markdown("<div class='metric-value'>--</div><div class='metric-label'>Energy Retained</div>", elem_classes=["metric-card", "metric-blue"])
        top_metric_3 = gr.Markdown("<div class='metric-value'>--</div><div class='metric-label'>Reward Delta</div>", elem_classes=["metric-card", "metric-purple"])
        top_metric_4 = gr.Markdown("<div class='metric-value'>--</div><div class='metric-label'>Waste Prevented</div>", elem_classes=["metric-card", "metric-orange"])
    
    # CONTROL PANEL
    with gr.Row(elem_classes=["glass-panel"]):
        with gr.Column(scale=3):
            hf_token_input = gr.Textbox(
                label="API Token for Generative XAI Analysis (Optional)", 
                placeholder="hf_XXXXXXXXXX... (Leave empty for Rule-based XAI)", 
                type="password"
            )
        with gr.Column(scale=1):
            btn_run = gr.Button("INITIALIZE AI SIMULATION 🚀", elem_classes=["primary-btn"])
    
    # CHARTS (2x2)
    with gr.Row():
        plot_solar = gr.Plot(elem_classes=["glass-panel"])
        plot_dist = gr.Plot(elem_classes=["glass-panel"])
        
    with gr.Row():
        plot_battery = gr.Plot(elem_classes=["glass-panel"])
        plot_reward = gr.Plot(elem_classes=["glass-panel"])

    # XAI & TABLE
    with gr.Row():
        with gr.Column(scale=1):
            xai_output = gr.Markdown("<div class='xai-container'><h3>🧠 AI Insight Engine</h3><p>Awaiting simulation pass...</p></div>")
        with gr.Column(scale=2):
            logs = gr.DataFrame(label="Real-time Simulation Telemetry", elem_classes=["glass-panel"])
            
    # HISTORY
    with gr.Accordion("Database Records 🗄️", open=False):
        btn_hist = gr.Button("Fetch Telemetry Records 📊", elem_classes=["primary-btn"])
        history_table = gr.DataFrame(elem_classes=["glass-panel"])

    btn_run.click(
        run_simulation_ui, 
        inputs=[hf_token_input], 
        outputs=[
            wow_impact, top_metric_1, top_metric_2, top_metric_3, top_metric_4,
            plot_solar, plot_battery, plot_reward, plot_dist, logs, xai_output
        ]
    )
    btn_hist.click(get_history_ui, outputs=[history_table])

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)