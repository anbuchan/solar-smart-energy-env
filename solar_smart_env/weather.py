import requests
import random
import time

# Open-Meteo API integration (No API Key Required)
# Coordinates for Bangalore, India
LAT = "12.9716"
LON = "77.5946"

_cached_weather = None
_last_fetch_time = 0

def get_weather_data():
    """
    Fetches real-time cloud coverage and shortwave radiation for Bangalore.
    Caches the response for 60 seconds to prevent API rate limiting.
    Returns:
        dict: {"cloud_cover": float (0.0 to 1.0), "radiation": float (W/m2)}
    """
    global _cached_weather, _last_fetch_time
    
    current_time = time.time()
    if _cached_weather is not None and (current_time - _last_fetch_time) < 60:
        return _cached_weather
        
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&current=cloud_cover,shortwave_radiation"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        current_data = data.get("current", {})
        cloud_cover = current_data.get("cloud_cover", 50) / 100.0
        radiation = current_data.get("shortwave_radiation", 500) # W/m²
        
        _cached_weather = {
            "cloud_cover": cloud_cover,
            "radiation": radiation
        }
        _last_fetch_time = current_time
        return _cached_weather
        
    except Exception as e:
        print(f"Weather API Warning: {e}")
        if _cached_weather is not None:
            return _cached_weather
        
        # Fallback simulated data
        return {
            "cloud_cover": round(random.uniform(0.0, 1.0), 2),
            "radiation": random.uniform(200, 800)
        }

def get_current_time_period():
    return random.choice(["morning", "afternoon", "night"])

