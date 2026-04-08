import math
import random
from models import EnergyObservation, EnergyAction
from weather import get_weather_data, get_current_time_period

class SolarEnergyEnv:
    def __init__(self, num_houses=4):
        self.num_houses = num_houses
        self.max_steps = 24  # 24 steps = 24 hours
        self.battery_capacity = 800.0  # Total capacity
        self.battery_health = 1.0  # Initial health 100%
        # Setup house profiles
        self.house_profiles = []
        for i in range(num_houses):
            # Each house has a base load, a peak hour, and a peak multiplier
            self.house_profiles.append({
                "base_load": random.uniform(10.0, 20.0),
                "peak_hour": random.choice([8, 12, 19, 21]), # Morning, Noon, Evening, Night
                "peak_multiplier": random.uniform(2.0, 4.0)
            })
        self.reset()

    def reset(self, task_name="easy", seed=None):
        if seed is not None:
            random.seed(seed)
            
        self.step_count = 0
        self.task_name = task_name
        self.cumulative_reward = 0.0
        self.wasted_energy = 0.0
        
        # Difficulty Modifiers
        if self.task_name == "hard":
            self.battery_charge = 50.0  
            self.demand_multiplier = 1.5
        elif self.task_name == "medium":
            self.battery_charge = 200.0
            self.demand_multiplier = 1.3
        else:
            self.battery_charge = 400.0 
            self.demand_multiplier = 1.0
            
        # Initial state fetch
        self.weather_data = get_weather_data()
        self.time_of_day = get_current_time_period()
        
        return self._get_obs()

    def _get_obs(self):
        # Time of day based on step 0-23
        hour = self.step_count % 24
        
        if 6 <= hour < 12:
            self.time_of_day = "morning"
        elif 12 <= hour < 18:
            self.time_of_day = "afternoon"
        elif 18 <= hour < 22:
            self.time_of_day = "evening"
        else:
            self.time_of_day = "night"

        # Smooth Solar curve (Sine wave over 12 daylight hours approx)
        # Sunrise around 6 AM, Sunset around 18 PM
        base_solar = 0.0
        if 6 <= hour <= 18:
            # hour=6 -> sin(0)=0. hour=12 -> sin(pi/2)=1. hour=18 -> sin(pi)=0
            rad = math.pi * ((hour - 6) / 12.0)
            radiation_max = self.weather_data.get("radiation", 500)
            base_solar = math.sin(rad) * (radiation_max * 0.8) # 80% efficiency of panels
        
        cloud_cover = self.weather_data.get("cloud_cover", 0.5)
        self.solar_generation = round(base_solar * (1.0 - (cloud_cover * 0.6)), 2)
        
        # Houses Demand Logic
        self.per_house_demand = {}
        total_demand = 0.0
        for i, profile in enumerate(self.house_profiles):
            # Base + random noise
            demand = profile["base_load"] + random.uniform(-2, 2)
            # Add peak if within 2 hours of peak hour
            if abs(hour - profile["peak_hour"]) <= 2:
                demand *= profile["peak_multiplier"]
            
            demand = round(demand * self.demand_multiplier, 2)
            self.per_house_demand[f"house_{i+1}"] = demand
            total_demand += demand
            
        self.total_demand = round(total_demand, 2)
        self.battery_efficiency = round(0.70 + (self.battery_health * 0.25), 2)

        # For compatibility with existing models, returning simple dict internally 
        # and letting wrapper handle arrays.
        self.current_state = {
            "step": self.step_count,
            "hour": hour,
            "time_of_day": self.time_of_day,
            "solar_generation": self.solar_generation,
            "total_demand": self.total_demand,
            "per_house_demand": self.per_house_demand,
            "battery_charge": round(self.battery_charge, 2),
            "battery_health": round(self.battery_health, 2),
            "battery_efficiency": self.battery_efficiency,
            "wasted_energy": round(self.wasted_energy, 2)
        }
        return self.current_state

    def step(self, action: int):
        """
        Actions: 
        0: Store energy (Solar -> Battery)
        1: Distribute energy (Solar -> Houses)
        2: Use Battery (Battery -> Houses)
        3: Reduce load (Simulation of demand response)
        """
        reward = 0
        
        # Degradation: Small decay every step
        self.battery_health = max(0.1, self.battery_health - 0.0005)
        
        available_solar = self.solar_generation
        demand_to_meet = self.total_demand
        
        self.per_house_distribution = {h: 0.0 for h in self.per_house_demand.keys()}
        
        if action == 0:  # STORE
            if available_solar > 0:
                charge_amt = available_solar * self.battery_efficiency
                if self.battery_charge + charge_amt <= self.battery_capacity:
                    self.battery_charge += charge_amt
                    reward += 0.5
                else:
                    overcharge = (self.battery_charge + charge_amt) - self.battery_capacity
                    self.battery_charge = self.battery_capacity
                    self.wasted_energy += overcharge
                    reward -= 0.5 # Wasting energy
            else:
                reward -= 1.0 # Invalid action at night
                
        elif action == 1:  # DISTRIBUTE SOLAR
            if available_solar > 0:
                ratio = min(1.0, available_solar / demand_to_meet)
                for h in self.per_house_demand:
                    self.per_house_distribution[h] = round(self.per_house_demand[h] * ratio, 2)
                
                if available_solar >= demand_to_meet:
                    reward += 2.0
                    excess = available_solar - demand_to_meet
                    # Auto-store excess as basic smart behavior
                    self.battery_charge = min(self.battery_capacity, self.battery_charge + (excess * self.battery_efficiency))
                else:
                    reward += 1.0 * ratio # Partial reward
            else:
                reward -= 1.0
                
        elif action == 2:  # USE BATTERY
            needed_battery = demand_to_meet / self.battery_efficiency
            if self.battery_charge >= needed_battery:
                self.battery_charge -= needed_battery
                for h in self.per_house_demand:
                    self.per_house_distribution[h] = self.per_house_demand[h]
                reward += 2.0
            else:
                ratio = self.battery_charge / needed_battery if needed_battery > 0 else 0
                for h in self.per_house_demand:
                    self.per_house_distribution[h] = round(self.per_house_demand[h] * ratio, 2)
                self.battery_charge = 0
                reward -= 0.5 # Blackout partial
                
        elif action == 3:  # REDUCE LOAD
            # Apply 30% blackout reduction
            for h in self.per_house_demand:
                reduced = self.per_house_demand[h] * 0.7
                self.per_house_distribution[h] = round(reduced, 2)
            self.total_demand *= 0.7
            reward += 0.2   # Survival but bad for users
            
        self.cumulative_reward += reward
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        info = {
            "per_house_distribution": self.per_house_distribution,
            "wasted_energy": self.wasted_energy
        }
        
        return self._get_obs(), round(reward, 2), done, info