import math
import random
from models import EnergyObservation, EnergyAction
from weather import get_weather_data

class SolarEnergyEnv:
    def __init__(self, num_houses=4):
        self.num_houses = num_houses
        self.max_steps = 24  
        self.battery_capacity = 150.0  
        self.battery_health = 1.0
        
        self.house_profiles = []
        for i in range(num_houses):
            self.house_profiles.append({
                "base_load": random.uniform(5.0, 10.0),
                "peak_hour": random.choice([8, 12, 19, 21]),
                "peak_multiplier": random.uniform(1.5, 2.5)
            })
        self.hospital_base_load = 30.0 
        self.reset()

    def reset(self, task_id="easy", seed=None, lat="12.9716", lon="77.5946"):
        if seed is not None:
            random.seed(seed)
            
        self.lat = lat
        self.lon = lon
        self.step_count = 0
        self.task_id = task_id
        self.cumulative_reward = 0.0
        self.total_demand_satisfied = 0.0
        self.total_hospital_satisfied = 0.0
        self.total_wasted_energy = 0.0
        
        if self.task_id == "hard":
            self.battery_charge = 10.0 
            self.demand_multiplier = 1.8 
            self.cloud_modifier = 0.8 
            self.hospital_active = True
        elif self.task_id == "medium":
            self.battery_charge = 50.0 
            self.demand_multiplier = 1.2
            self.cloud_modifier = 0.4
            self.hospital_active = False 
        else:
            self.battery_charge = 150.0 
            self.demand_multiplier = 0.8 
            self.cloud_modifier = 0.0 
            self.hospital_active = False
            
        self.weather_data = get_weather_data(lat=self.lat, lon=self.lon)
        self._update_state()
        return self.state()

    def state(self):
        return {
            "step": self.step_count,
            "hour": self.step_count % 24,
            "time_of_day": self.time_of_day,
            "solar_generation": round(self.solar_generation, 2),
            "total_demand": round(self.total_demand, 2),
            "per_house_demand": {h: round(v, 2) for h, v in self.per_house_demand.items()},
            "hospital_demand": round(self.hospital_demand, 2),
            "battery_charge": round(self.battery_charge, 2),
            "battery_soc": round(self.battery_charge / self.battery_capacity, 2),
            "battery_health": round(self.battery_health, 2),
            "grid_price": 0.15,
            "is_raining": self.weather_data.get("cloud_cover", 0) > 0.7
        }

    def _update_state(self):
        hour = self.step_count % 24
        if 6 <= hour < 12: self.time_of_day = "morning"
        elif 12 <= hour < 18: self.time_of_day = "afternoon"
        elif 18 <= hour < 22: self.time_of_day = "evening"
        else: self.time_of_day = "night"

        base_solar = 0.0
        if 6 <= hour <= 18:
            rad = math.pi * ((hour - 6) / 12.0)
            base_solar = math.sin(rad) * 500
        
        cloud_cover = min(1.0, self.weather_data.get("cloud_cover", 0.5) + self.cloud_modifier)
        self.solar_generation = max(0.0, base_solar * (1.0 - (cloud_cover * 0.7)))
        
        self.per_house_demand = {}
        total_home_demand = 0.0
        for i, profile in enumerate(self.house_profiles):
            demand = profile["base_load"]
            if abs(hour - profile["peak_hour"]) <= 2:
                demand *= profile["peak_multiplier"]
            h_demand = demand * self.demand_multiplier
            self.per_house_demand[f"house_{i+1}"] = h_demand
            total_home_demand += h_demand
            
        self.hospital_demand = self.hospital_base_load if self.hospital_active else 0.0
        self.total_demand = total_home_demand + self.hospital_demand

    def calculate_grader_score(self):
        """
        Deterministic grader as required by OpenEnv spec.
        Weights: Demand Satisfaction (40%), Critical Handling (40%), Efficiency (20%)
        Returns: float (0.0 to 1.0)
        """
        if self.step_count == 0: return 0.0
        
        # 1. Demand Satisfaction for Homes
        home_sat = min(1.0, self.total_demand_satisfied / (max(1.0, self.total_demand - (self.hospital_demand * self.step_count))))
        
        # 2. Critical Zone Handling
        crit_sat = 1.0
        if self.hospital_active:
            crit_sat = min(1.0, self.total_hospital_satisfied / (self.hospital_base_load * self.step_count))
            
        # 3. Efficiency (Lower waste is better)
        efficiency = max(0.0, 1.0 - (self.total_wasted_energy / 500.0)) # Scaled against 500kWh waste threshold
        
        final_score = (0.4 * home_sat) + (0.4 * crit_sat) + (0.2 * efficiency)
        return round(float(final_score), 2)

    def close(self):
        """
        Cleanup logic as required by OpenEnv lifecycle.
        """
        pass
    
    def step(self, action: int):
        available_energy = self.solar_generation + self.battery_charge
        energy_to_homes = 0.0
        energy_to_hospital = 0.0
        energy_to_battery = 0.0
        wasted_energy = 0.0
        
        target_hospital = self.hospital_demand
        target_homes = sum(self.per_house_demand.values())
        
        if action == 0: # STORE
            energy_to_battery = min(self.solar_generation, self.battery_capacity - self.battery_charge)
            remaining = self.solar_generation - energy_to_battery
            energy_to_hospital = min(remaining, target_hospital)
            remaining -= energy_to_hospital
            energy_to_homes = min(remaining, target_homes)
            wasted_energy = remaining - energy_to_homes
        elif action == 1: # DISTRIBUTE
            energy_to_hospital = min(available_energy, target_hospital)
            available_energy -= energy_to_hospital
            energy_to_homes = min(available_energy, target_homes)
            available_energy -= energy_to_homes
            if available_energy > self.battery_charge:
                excess = self.solar_generation - (energy_to_hospital + energy_to_homes)
                energy_to_battery = min(excess, self.battery_capacity - self.battery_charge)
                wasted_energy = excess - energy_to_battery
            else:
                energy_to_battery = available_energy - self.battery_charge
        elif action == 2: # REDUCE LOAD
            target_homes *= 0.5 
            energy_to_hospital = min(available_energy, target_hospital)
            available_energy -= energy_to_hospital
            energy_to_homes = min(available_energy, target_homes)
            available_energy -= energy_to_homes
            if available_energy > self.battery_charge:
                energy_to_battery = min(self.solar_generation - (energy_to_hospital + energy_to_homes), self.battery_capacity - self.battery_charge)
                wasted_energy = self.solar_generation - (energy_to_hospital + energy_to_homes) - energy_to_battery
            else:
                energy_to_battery = available_energy - self.battery_charge
        elif action == 3: # PRIORITIZE CRITICAL
            energy_to_hospital = min(available_energy, target_hospital)
            available_energy -= energy_to_hospital
            energy_to_homes = 0.0
            if available_energy > self.battery_charge:
                 energy_to_battery = min(self.solar_generation - energy_to_hospital, self.battery_capacity - self.battery_charge)
                 wasted_energy = self.solar_generation - energy_to_hospital - energy_to_battery
            else:
                 energy_to_battery = available_energy - self.battery_charge

        self.battery_charge = max(0.0, min(self.battery_capacity, self.battery_charge + energy_to_battery))
        self.total_demand_satisfied += energy_to_homes
        self.total_hospital_satisfied += energy_to_hospital
        self.total_wasted_energy += wasted_energy
        
        # Reward Function (Dynamic Feedback)
        step_reward = 0.0
        if target_hospital > 0 and energy_to_hospital < target_hospital:
            step_reward = -1.0
        else:
            sat_ratio = (energy_to_hospital + energy_to_homes) / (self.total_demand + 0.01)
            step_reward = 1.0 if sat_ratio > 0.9 else (0.6 if sat_ratio > 0.6 else 0.2)
            if wasted_energy > 5.0: step_reward -= 0.1
            
        step_reward = round(float(step_reward), 2)
        self.cumulative_reward += step_reward
        self.step_count += 1
        
        done = self.step_count >= self.max_steps
        if self.battery_charge <= 0.1 and self.task_id == "hard": done = True
            
        self._update_state()
        return self.state(), step_reward, done, {"score": self.calculate_grader_score()}