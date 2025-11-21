import argparse
import os
import sys
import subprocess

def run_dashboard():
    print("Starting AgriSense Dashboard...")
    # Use sys.executable to ensure we use the current python environment
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"])

def run_cli():
    print("Running in CLI mode...")
    from src.pipeline.runner import run_pipeline
    # Example ROI
    roi = [
        [-120.1, 36.1],
        [-120.1, 36.2],
        [-120.0, 36.2],
        [-120.0, 36.1],
        [-120.1, 36.1]
    ]
    run_pipeline(roi)

def run_api():
    print("Starting AgriSense API Backend...")
    subprocess.run([sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriSense Entry Point")
    parser.add_argument("--mode", choices=["dashboard", "cli", "api"], default="dashboard", help="Mode to run the application in.")
    
    args = parser.parse_args()
    
    if args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "api":
        run_api()
    else:
        run_cli()
