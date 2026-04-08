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
        self.total_demand_accumulated = 0.0
        self.total_hospital_demand_accumulated = 0.0

        if self.task_id == "hard":
            self.battery_charge = 40.0
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
        return self.state(), {"score": self.calculate_grader_score()}

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
            "is_raining": self.weather_data.get("cloud_cover", 0) > 0.7,
            "per_house_distribution": {h: round(v, 2) for h, v in getattr(self, "per_house_distribution", {}).items()}
        }

    def _update_state(self):
        hour = self.step_count % 24

        if 6 <= hour < 12:
            self.time_of_day = "morning"
        elif 12 <= hour < 18:
            self.time_of_day = "afternoon"
        elif 18 <= hour < 22:
            self.time_of_day = "evening"
        else:
            self.time_of_day = "night"

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

        # Add noise
        self.total_demand += random.uniform(-3, 3)
        self.solar_generation += random.uniform(-10, 10)
        self.solar_generation = max(0.0, self.solar_generation)

        self.total_demand_accumulated += self.total_demand
        self.total_hospital_demand_accumulated += self.hospital_demand

    # ✅ FINAL FIXED GRADER
    def calculate_grader_score(self):
        if self.step_count == 0:
            return 0.05

        total_home_demand = max(1.0, self.total_demand_accumulated)
        home_sat = min(1.0, self.total_demand_satisfied / total_home_demand)

        crit_sat = 1.0
        if self.hospital_active:
            total_crit = max(1.0, self.total_hospital_demand_accumulated)
            crit_sat = min(1.0, self.total_hospital_satisfied / total_crit)

        efficiency = max(0.0, 1.0 - (self.total_wasted_energy / 500.0))

        score = (0.4 * home_sat) + (0.4 * crit_sat) + (0.2 * efficiency)

        if math.isnan(score) or math.isinf(score):
            score = 0.5

        # 🚨 CRITICAL FIX
        score = round(float(score), 2)

        if score >= 1.0:
            score = 0.99
        elif score <= 0.0:
            score = 0.01

        return score

    def close(self):
        pass

    def step(self, action: int):
        available_energy = self.solar_generation + self.battery_charge

        energy_to_homes = 0.0
        energy_to_hospital = 0.0
        energy_to_battery = 0.0
        wasted_energy = 0.0

        target_hospital = self.hospital_demand
        target_homes = sum(self.per_house_demand.values())

        if action == 0:
            energy_to_battery = min(self.solar_generation, self.battery_capacity - self.battery_charge)
        elif action == 1:
            energy_to_hospital = min(available_energy, target_hospital)
            available_energy -= energy_to_hospital
            energy_to_homes = min(available_energy, target_homes)
        elif action == 2:
            target_homes *= 0.5
            energy_to_hospital = min(available_energy, target_hospital)
            available_energy -= energy_to_hospital
            energy_to_homes = min(available_energy, target_homes)
        elif action == 3:
            energy_to_hospital = min(available_energy, target_hospital)

        # Battery update
        self.battery_charge += energy_to_battery
        self.battery_charge = max(0.0, min(self.battery_capacity, self.battery_charge))

        self.total_demand_satisfied += energy_to_homes
        self.total_hospital_satisfied += energy_to_hospital
        self.total_wasted_energy += wasted_energy

        # Distribution tracking
        self.per_house_distribution = {}
        total_home_demand = sum(self.per_house_demand.values())

        if total_home_demand > 0:
            ratio = energy_to_homes / total_home_demand
            for h, d in self.per_house_demand.items():
                self.per_house_distribution[h] = d * ratio
        else:
            self.per_house_distribution = {h: 0.0 for h in self.per_house_demand}

        # Reward
        sat_ratio = (energy_to_hospital + energy_to_homes) / (self.total_demand + 0.01)

        if sat_ratio > 0.95:
            step_reward = 0.95
        elif sat_ratio > 0.75:
            step_reward = 0.7
        elif sat_ratio > 0.5:
            step_reward = 0.4
        else:
            step_reward = -0.5

        if wasted_energy > 5.0:
            step_reward -= 0.1

        step_reward = round(float(step_reward), 2)

        self.cumulative_reward += step_reward
        self.step_count += 1

        done = self.step_count >= self.max_steps

        self._update_state()

        return self.state(), step_reward, done, {
            "score": self.calculate_grader_score(),
            "battery": self.battery_charge,
            "solar": self.solar_generation,
            "per_house_distribution": {h: round(v, 2) for h, v in self.per_house_distribution.items()}
        }
