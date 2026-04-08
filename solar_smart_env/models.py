from pydantic import BaseModel
from typing import Dict
from enum import Enum

class EnergyAction(str, Enum):
    store_energy = "store_energy"
    distribute_energy = "distribute_energy"
    reduce_load = "reduce_load"
    prioritize_critical = "prioritize_critical"

class EnergyObservation(BaseModel):
    step: int
    hour: int
    time_of_day: str
    solar_generation: float
    total_demand: float
    battery_charge: float
    battery_soc: float
    battery_health: float
    grid_price: float
    is_raining: bool
    per_house_demand: Dict[str, float]
    hospital_demand: float