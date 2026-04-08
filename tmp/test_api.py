import requests
import time
import subprocess
import os
import sys

def test_api():
    print("Starting API test...")
    # Start server in background using venv
    server_process = subprocess.Popen(
        [os.path.join(".venv", "Scripts", "python.exe"), "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "7861"],
        cwd=os.getcwd()
    )
    
    time.sleep(5) # Wait for server to start
    
    try:
        # Test /reset
        print("Testing /reset...")
        r = requests.post("http://127.0.0.1:7861/reset", json={"task_id": "medium"})
        r.raise_for_status()
        data = r.json()
        assert "observation" in data, "observation missing from /reset response"
        assert "info" in data, "info missing from /reset response"
        assert "score" in data["info"], "score missing from /reset info"
        print(f"Reset Score: {data['info']['score']}")
        
        # Test /step
        print("Testing /step...")
        r = requests.post("http://127.0.0.1:7861/step", json={"action": 1})
        r.raise_for_status()
        data = r.json()
        assert "observation" in data, "observation missing from /step"
        assert "reward" in data, "reward missing from /step"
        assert "done" in data, "done missing from /step"
        assert "info" in data, "info missing from /step"
        assert "score" in data["info"], "score missing from /step info"
        print(f"Step Score: {data['info']['score']}, Reward: {data['reward']}")
        
        print("API tests passed!")
    finally:
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    test_api()
