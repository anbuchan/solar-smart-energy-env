import sqlite3
import os
import json

DB_PATH = "solar_simulation.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS simulation_history_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            step INTEGER,
            action INTEGER,
            solar_generation REAL,
            battery_level REAL,
            total_demand REAL,
            per_house_demand TEXT,
            per_house_distribution TEXT,
            reward REAL,
            baseline_reward REAL,
            efficiency REAL,
            wasted_energy REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_step(step, action, solar, battery, total_demand, per_house_demand, per_house_distribution, reward, baseline_reward, efficiency, wasted_energy, timestamp=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if timestamp:
        cursor.execute('''
            INSERT INTO simulation_history_v2 (
                step, action, solar_generation, battery_level, total_demand, 
                per_house_demand, per_house_distribution, reward, baseline_reward, efficiency, wasted_energy, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            step, action, solar, battery, total_demand, 
            json.dumps(per_house_demand), json.dumps(per_house_distribution), 
            reward, baseline_reward, efficiency, wasted_energy, timestamp
        ))
    else:
        cursor.execute('''
            INSERT INTO simulation_history_v2 (
                step, action, solar_generation, battery_level, total_demand, 
                per_house_demand, per_house_distribution, reward, baseline_reward, efficiency, wasted_energy
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            step, action, solar, battery, total_demand, 
            json.dumps(per_house_demand), json.dumps(per_house_distribution), 
            reward, baseline_reward, efficiency, wasted_energy
        ))
    conn.commit()
    conn.close()

def get_history(limit=100):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM simulation_history_v2 ORDER BY timestamp DESC, id ASC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    
    col_names = [description[0] for description in cursor.description]
    
    result = []
    for row in rows:
        row_dict = dict(zip(col_names, row))
        try:
            d1 = json.loads(row_dict['per_house_demand'])
            row_dict['per_house_demand'] = str(d1)
            
            d2 = json.loads(row_dict['per_house_distribution'])
            row_dict['per_house_distribution'] = str(d2)
        except:
            pass
        result.append(row_dict)
        
    conn.close()
    return result

if __name__ == "__main__":
    init_db()
