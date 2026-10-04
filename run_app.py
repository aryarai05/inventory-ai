"""
Optix Enterprise Application Launcher
Starts the production FastAPI server hosting the modern web UI and REST endpoints.
"""

import sys
import os
import webbrowser
import threading
import time
import uvicorn

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def open_browser():
    """Waits for server to spin up then opens default browser."""
    time.sleep(1.2)
    webbrowser.open("http://localhost:8000")


def main():
    print("\n" + "=" * 75)
    print("      OPTIX // ENTERPRISE INVENTORY AI & REPLENISHMENT PLATFORM      ")
    print("=" * 75)
    print("  * Modern Web UI:  http://localhost:8000")
    print("  * REST API /Docs: http://localhost:8000/docs")
    print("  * Local Solver:   PuLP CBC (Mixed-Integer Linear Programming)")
    print("  * Primary Model:  LightGBM Regressor (Tuned)")
    print("=" * 75 + "\n")

    # Start browser opener in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Launch server
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")


if __name__ == "__main__":
    main()
