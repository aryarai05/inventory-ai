"""
End-to-End Orchestrator Pipeline
Executes the full pipeline sequentially:
1. Synthetic Data Generation (3 years, 50 SKUs, 5 categories, 5 regions)
2. Database initialization and feature engineering without leakage
3. Baseline and ML model training with LightGBM tuning and MLflow logging
4. Forward forecasting and PuLP mathematical replenishment optimization
"""

import sys
import time
import subprocess


def run_step(step_name: str, script_name: str):
    print("\n" + "=" * 75)
    print(f"  RUNNING PIPELINE STEP: {step_name} ({script_name})")
    print("=" * 75)
    start_t = time.time()
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    elapsed = time.time() - start_t
    if result.returncode != 0:
        print(f"\n[!] Pipeline failed at step: {step_name} with return code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[+] {step_name} completed in {elapsed:.1f} seconds.")


def main():
    total_start = time.time()
    print("===========================================================================")
    print("      AI-POWERED INVENTORY DEMAND FORECASTING & OPTIMIZATION PIPELINE      ")
    print("===========================================================================")

    run_step("1. Synthetic Data Generation", "generate_data.py")
    run_step("2. SQL Database & Feature Pipeline", "preprocess.py")
    run_step("3. Forecasting Models & MLflow Tracking", "train.py")
    run_step("4. Inventory Risk & PuLP Optimization", "optimize.py")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 75)
    print(f"  FULL END-TO-END PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed:.1f} SECONDS")
    print("=" * 75)
    print("  To launch the enterprise decision platform, run:")
    print("  python run_app.py")
    print("=" * 75)


if __name__ == "__main__":
    main()
