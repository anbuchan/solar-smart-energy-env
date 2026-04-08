from pydantic import BaseModel
from typing import Optional

class EnergyObservation(BaseModel):
    step: int
    time_of_day: str
    weather: str
    cloud_coverage: float
    solar_generation: float
    total_demand: float
    battery_charge: float
    battery_health: float
    battery_efficiency: float

class EnergyAction(BaseModel):
    # 0: store, 1: distribute, 2: use_battery, 3: reduce_load
    action: int