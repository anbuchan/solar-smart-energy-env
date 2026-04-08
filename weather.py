import requests
import random
import time

# Open-Meteo API integration (No API Key Required)
# Coordinates for Bangalore, India
LAT = "12.9716"
LON = "77.5946"

_cached_weather = None
_last_fetch_time = 0

def get_location_coords(query: str):
    """
    Uses Open-Meteo Geocoding API to find coordinates for a city/location.
    Intelligently splits "City, Country" queries to improve search success.
    """
    if not query or len(query.strip()) < 2:
        return LAT, LON, "Bangalore, India (Default)"
        
    try:
        parts = [p.strip() for p in query.split(",")]
        search_name = parts[0]
        country_filter = parts[1].lower() if len(parts) > 1 else None
        
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": search_name,
            "count": 10, # Fetch more to find the right country
            "language": "en",
            "format": "json"
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        results = data.get("results", [])
        if not results:
            return LAT, LON, "Bangalore, India (Fallback)"
            
        # Try to find a match that includes the country if specified
        target_res = results[0]
        if country_filter:
            for res in results:
                res_country = res.get("country", "").lower()
                if country_filter in res_country:
                    target_res = res
                    break
        
        name = f"{target_res.get('name')}, {target_res.get('country')}"
        return str(target_res.get("latitude")), str(target_res.get("longitude")), name
        
    except Exception as e:
        print(f"Geocoding Error: {e}")
        
    return LAT, LON, "Bangalore, India (Fallback)"

def get_weather_data(lat=LAT, lon=LON):
    """
    Fetches real-time cloud coverage and shortwave radiation for specific coordinates.
    Caches the response for 60 seconds to prevent API rate limiting.
    Returns:
        dict: {"cloud_cover": float (0.0 to 1.0), "radiation": float (W/m2)}
    """
    global _cached_weather, _last_fetch_time
    
    current_time = time.time()
    # Cache key includes lat/lon to prevent cross-location cache issues
    cache_key = f"{lat}_{lon}"
    
    if _cached_weather is not None and _cached_weather.get("key") == cache_key and (current_time - _last_fetch_time) < 60:
        return _cached_weather
        
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=cloud_cover,shortwave_radiation"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        current_data = data.get("current", {})
        cloud_cover = current_data.get("cloud_cover", 50) / 100.0
        radiation = current_data.get("shortwave_radiation", 500) # W/m²
        
        _cached_weather = {
            "key": cache_key,
            "cloud_cover": cloud_cover,
            "radiation": radiation
        }
        _last_fetch_time = current_time
        return _cached_weather
        
    except Exception as e:
        print(f"Weather API Warning: {e}")
        if _cached_weather is not None and _cached_weather.get("key") == cache_key:
            return _cached_weather
        
        # Fallback simulated data
        return {
            "cloud_cover": round(random.uniform(0.0, 1.0), 2),
            "radiation": random.uniform(200, 800)
        }

def get_current_time_period():
    return random.choice(["morning", "afternoon", "night"])

