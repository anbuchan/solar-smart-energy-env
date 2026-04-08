from fastapi import FastAPI
from env import SolarEnergyEnv
from rl_agent import SolarGymEnv, get_trained_model
from database import save_step, get_history, init_db
from llm import generate_xai_report
from weather import get_location_coords
from models import EnergyObservation
from datetime import datetime
import uvicorn
import gradio as gr
import pandas as pd
import json
import random
import plotly.graph_objects as go

app = FastAPI()

# Initialization on Startup
try:
    init_db()
except Exception as e:
    print(f"Database initialization error: {e}")

@app.get("/")
def read_root():
    return {"message": "Solar Smart Energy API is active ☀️"}

@app.get("/history")
def history():
    try:
        data = get_history()
        return {"history": data}
    except Exception as e:
        return {"history": [], "error": str(e)}

# Global Environment State
shared_env = SolarEnergyEnv()

@app.post("/reset")
def reset_env(data: dict = None):
    task_id = data.get("task_id", data.get("task_name", "easy")) if data else "easy"
    obs, info = shared_env.reset(task_id=task_id)
    return {"observation": obs, "info": info}

@app.post("/step")
def step_env(data: dict):
    action = data.get("action", 1)
    obs, reward, done, info = shared_env.step(action)
    return {
        "observation": obs,
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }

@app.get("/state", response_model=EnergyObservation)
def get_state():
    return shared_env.state()

def format_action(action):
    mapping = {0: "Store 🔋", 1: "Distribute 🏠", 2: "Reduce Load 🚨", 3: "Prioritize Critical 🏥"}
    return mapping.get(action, "Unknown")

def create_plotly_figure(df):
    bg_color = "rgba(10,10,10,0.6)"
    grid_color = "rgba(255,255,255,0.05)"
    text_color = "#ffffff"
    
    fig_solar = go.Figure()
    fig_solar.add_trace(go.Scatter(x=df['Step'], y=df['Solar'], name="Solar Generation", line=dict(color="#00ffcc", width=4, shape='spline'), fill='tozeroy', fillcolor="rgba(0, 255, 204, 0.15)"))
    fig_solar.add_trace(go.Scatter(x=df['Step'], y=df['Demand'], name="Total Demand", line=dict(color="#ff0055", width=3, shape='spline', dash='dot')))
    fig_solar.add_trace(go.Scatter(x=df['Step'], y=df['Hospital_Demand'], name="Critical Load (Hospital)", line=dict(color="#ef4444", width=3, shape='spline')))
    fig_solar.update_layout(title="⚡ Solar Input vs Grid Demand", paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=text_color), hovermode="x unified", xaxis=dict(gridcolor=grid_color), yaxis=dict(gridcolor=grid_color), margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    fig_battery = go.Figure()
    fig_battery.add_trace(go.Scatter(x=df['Step'], y=df['Battery'], name="Battery Charge", line=dict(color="#00ccff", width=4, shape='spline'), fill='tozeroy', fillcolor="rgba(0, 204, 255, 0.2)"))
    fig_battery.update_yaxes(range=[0, 160])
    fig_battery.update_layout(title="🔋 Energy Storage (kWh) Limit: 150", paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=text_color), hovermode="x unified", xaxis=dict(gridcolor=grid_color), yaxis=dict(gridcolor=grid_color), margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    fig_reward = go.Figure()
    fig_reward.add_trace(go.Scatter(x=df['Step'], y=df['Reward'], name="PPO AI Agent", line=dict(color="#b026ff", width=4, shape='spline')))
    fig_reward.add_trace(go.Scatter(x=df['Step'], y=df['Baseline_Reward'], name="Random Baseline", line=dict(color="#64748b", width=2, dash="dash", shape='spline')))
    fig_reward.update_layout(title="🤖 AI Performance Tracking (Reward)", paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=text_color), hovermode="x unified", xaxis=dict(gridcolor=grid_color), yaxis=dict(gridcolor=grid_color), margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    fig_dist = go.Figure()
    if len(df) > 0 and 'Raw_Distribution' in df.columns:
        houses = list(df.iloc[0]['Raw_Distribution'].keys())
        colors = ["#ec4899", "#8b5cf6", "#14b8a6", "#f97316", "#eab308"]
        for i, house in enumerate(houses):
            y_vals = [d.get(house, 0) for d in df['Raw_Distribution']]
            fig_dist.add_trace(go.Scatter(x=df['Step'], y=y_vals, name=house, line=dict(width=2, color=colors[i%len(colors)], shape='spline'), stackgroup='one', fillcolor=colors[i%len(colors)]))
    fig_dist.update_layout(title="🏠 Per-House Energy Allocation", paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=text_color), hovermode="x unified", xaxis=dict(gridcolor=grid_color), yaxis=dict(gridcolor=grid_color), margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    return fig_solar, fig_battery, fig_reward, fig_dist

def run_simulation_ui(hf_token, task_type="easy", location_query=""):
    try:
        lat, lon, resolved_name = get_location_coords(location_query)
        sim_seed = random.randint(1, 1000)
        
        rl_sim = SolarGymEnv()
        obs_dict_rl, _ = rl_sim.reset(seed=sim_seed, task_id=task_type, lat=lat, lon=lon)
        
        base_sim = SolarGymEnv()
        base_sim.reset(seed=sim_seed, task_id=task_type, lat=lat, lon=lon)
        
        agent_model = get_trained_model()
            
        data = []
        common_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        current_obs_rl = obs_dict_rl
        for step_idx in range(24):
            action, _ = agent_model.predict(current_obs_rl, deterministic=True)
            action = int(action)
            new_obs_norm_rl, rl_reward, done, _, info_rl = rl_sim.step(action)
            current_obs_rl = new_obs_norm_rl
            
            random_action = base_sim.action_space.sample()
            _, base_reward, _, _, _ = base_sim.step(random_action)
            
            # Get raw env state for metrics
            env_state = rl_sim.env.state()
            wasted_ai = info_rl.get("wasted_energy", 0)
            
            try:
                save_step(
                    step=step_idx, action=action, solar=env_state["solar_generation"], 
                    battery=env_state["battery_charge"], total_demand=env_state["total_demand"], 
                    per_house_demand=env_state["per_house_demand"], 
                    per_house_distribution=info_rl.get("per_house_distribution", {}), 
                    reward=rl_reward, baseline_reward=base_reward,
                    efficiency=env_state["battery_soc"], wasted_energy=wasted_ai,
                    timestamp=common_timestamp
                )
            except: pass
            
            # Formatted list for UI Table
            dist_summary = f"🏥: {env_state['hospital_demand']:.1f} | " + ", ".join([f"{k}: {v:.1f}" for k, v in env_state["per_house_demand"].items()])
            
            data.append({
                "Step": step_idx, "Action": format_action(action), "Solar": env_state["solar_generation"],
                "Battery": env_state["battery_charge"], "Demand": env_state["total_demand"],
                "Hospital_Demand": env_state["hospital_demand"],
                "Reward": rl_reward, "Baseline_Reward": base_reward, "Wasted": wasted_ai,
                "Distribution": dist_summary,
                "Raw_Distribution": env_state["per_house_demand"]
            })
            if done: break
            
        df = pd.DataFrame(data)
        fig_plots = create_plotly_figure(df)
        
        total_ai_reward = df["Reward"].sum()
        total_base_reward = df["Baseline_Reward"].sum()
        reward_diff = total_ai_reward - total_base_reward
        eff_gain = (reward_diff / abs(total_base_reward + 0.1)) * 100 if total_base_reward != 0 else 100.0
        eff_gain_str = f"+{eff_gain:.1f}%" if eff_gain >= 0 else f"{eff_gain:.1f}%"
        
        impact_statement = f"<div class='wow-impact'>🚀 Simulation: {task_type.upper()} MODE in {resolved_name}. Logic Efficiency: {eff_gain_str}</div>"
        m1 = f"<div class='metric-value'>{eff_gain_str}</div><div class='metric-label'>Efficiency Gain</div>"
        m2 = f"<div class='metric-value'>{df['Battery'].iloc[-1]:.1f} kWh</div><div class='metric-label'>Energy Retained</div>"
        m3 = f"<div class='metric-value'>+{reward_diff:.1f}</div><div class='metric-label'>AI Reward Delta</div>"
        m4 = f"<div class='metric-value'>{max(0, 100.0 - (df['Wasted'].mean()*2)):.1f}%</div><div class='metric-label'>Waste Prevented</div>"
        
        xai_audit = f"<div class='xai-container'><h3>🧠 AI Insight Engine</h3><p>{generate_xai_report(df, hf_token)}</p></div>"
        
        # UI DataFrame
        display_df = df.drop(columns=["Raw_Distribution"])
        
        return [impact_statement, m1, m2, m3, m4] + list(fig_plots) + [display_df, xai_audit]
        
    except Exception as e:
        err_msg = f"<div class='wow-impact' style='background:rgba(255,0,0,0.1); border-color:red;'>⚠️ Simulation Error: {e}</div>"
        return [err_msg] + ["--"]*4 + [None]*4 + [pd.DataFrame(), str(e)]

def get_history_ui():
    try:
        history = get_history()
        return pd.DataFrame(history) if history else pd.DataFrame()
    except:
        return pd.DataFrame()

# UI Styles
custom_css = """
body, .gradio-container { background-color: #0a0a0a !important; background-image: radial-gradient(circle at top, #111827 0%, #0a0a0a 100%) !important; color: #ffffff !important; font-family: 'Inter', sans-serif !important; }
.glass-panel { background: rgba(20, 20, 20, 0.75) !important; backdrop-filter: blur(15px) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; border-radius: 16px !important; padding: 20px !important; margin-bottom: 20px !important; }
.neon-title { text-align: center; font-weight: 900 !important; font-size: 3em !important; text-shadow: 0 0 20px rgba(0, 255, 204, 0.5) !important; }
.neon-subtitle { text-align: center; color: #00ccff !important; font-size: 1.2em; text-transform: uppercase; letter-spacing: 4px; }
.metric-card { background: rgba(30,30,30,0.8) !important; border-radius: 20px !important; padding: 30px 10px !important; text-align: center !important; }
.metric-green { border-bottom: 4px solid #00ffcc !important; } .metric-blue { border-bottom: 4px solid #00ccff !important; } .metric-purple { border-bottom: 4px solid #b026ff !important; } .metric-orange { border-bottom: 4px solid #ffaa00 !important; }
.metric-value { font-size: 2.8em; font-weight: 900; } .metric-label { font-size: 0.95em; color: #a1a1aa; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css) as demo:
    gr.Markdown("<h1 class='neon-title'>⚡ AI Smart Solar Command Center</h1>")
    gr.Markdown("<div class='neon-subtitle'>Production Autonomous Grid Management</div>")
    
    wow_impact = gr.Markdown("<div class='wow-impact'>Awaiting Simulation...</div>")
    
    with gr.Row():
        top_metric_1 = gr.Markdown("<div class='metric-value'>--</div><div class='metric-label'>Efficiency Gain</div>", elem_classes=["metric-card", "metric-green"])
        top_metric_2 = gr.Markdown("<div class='metric-value'>--</div><div class='metric-label'>Energy Retained</div>", elem_classes=["metric-card", "metric-blue"])
        top_metric_3 = gr.Markdown("<div class='metric-value'>--</div><div class='metric-label'>Reward Delta</div>", elem_classes=["metric-card", "metric-purple"])
        top_metric_4 = gr.Markdown("<div class='metric-value'>--</div><div class='metric-label'>Waste Prevented</div>", elem_classes=["metric-card", "metric-orange"])
    
    with gr.Row(elem_classes=["glass-panel"]):
        location_input = gr.Textbox(label="🌐 Geographic Targeting", value="Bangalore, India")
        hf_token_input = gr.Textbox(label="🔑 AI Analysis Token (Optional)", type="password")
            
    with gr.Row():
        btn_easy = gr.Button("🟢 START EASY TASK", variant="primary")
        btn_medium = gr.Button("🟡 START MEDIUM TASK", variant="secondary")
        btn_hard = gr.Button("🔴 START HARD TASK", variant="stop")
    
    with gr.Row():
        plot_solar = gr.Plot(elem_classes=["glass-panel"])
        plot_dist = gr.Plot(elem_classes=["glass-panel"])
    with gr.Row():
        plot_battery = gr.Plot(elem_classes=["glass-panel"])
        plot_reward = gr.Plot(elem_classes=["glass-panel"])

    with gr.Row():
        xai_output = gr.Markdown("<div class='xai-container'><h3>🧠 AI Insight Engine</h3><p>Awaiting simulation pass...</p></div>")
        logs = gr.DataFrame(label="Simulation Telemetry")
            
    with gr.Accordion("Database Records 🗄️", open=False):
        btn_hist = gr.Button("Refresh History 📊")
        history_table = gr.DataFrame()

    difficulty_state = gr.State("easy")

    btn_easy.click(lambda: "easy", outputs=difficulty_state).then(
        run_simulation_ui, inputs=[hf_token_input, difficulty_state, location_input], 
        outputs=[wow_impact, top_metric_1, top_metric_2, top_metric_3, top_metric_4, plot_solar, plot_battery, plot_reward, plot_dist, logs, xai_output]
    )
    btn_medium.click(lambda: "medium", outputs=difficulty_state).then(
        run_simulation_ui, inputs=[hf_token_input, difficulty_state, location_input], 
        outputs=[wow_impact, top_metric_1, top_metric_2, top_metric_3, top_metric_4, plot_solar, plot_battery, plot_reward, plot_dist, logs, xai_output]
    )
    btn_hard.click(lambda: "hard", outputs=difficulty_state).then(
        run_simulation_ui, inputs=[hf_token_input, difficulty_state, location_input], 
        outputs=[wow_impact, top_metric_1, top_metric_2, top_metric_3, top_metric_4, plot_solar, plot_battery, plot_reward, plot_dist, logs, xai_output]
    )
    btn_hist.click(get_history_ui, outputs=[history_table])

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
