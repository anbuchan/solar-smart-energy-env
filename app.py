import uvicorn
import os

# Legacy Root Entry Point Fallback
# This file ensures that any validator checking for '/' or 'app.py' in the root
# is correctly directed to the new modular 'server/app.py' logic.

if __name__ == "__main__":
    print("Starting Solar Smart Grid (Root Fallback Mode)...")
    # Redirect to the server.app logic
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)
