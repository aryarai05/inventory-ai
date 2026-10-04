"""
Generate Synthetic Dataset Script
Entrypoint for generating the 3-year supply chain and inventory historical dataset.
"""

import sys
import os
import yaml

# Add root directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.generator import SyntheticInventoryDataGenerator


def main():
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        start_date = config["data"]["start_date"]
        end_date = config["data"]["end_date"]
        raw_dir = config["data"]["raw_dir"]
        seed = config["project"]["random_seed"]
    else:
        start_date = "2023-01-01"
        end_date = "2025-12-31"
        raw_dir = "data/raw"
        seed = 42

    print(f"[*] Starting realistic dataset generation from {start_date} to {end_date} (Seed: {seed})...")
    generator = SyntheticInventoryDataGenerator(
        start_date=start_date,
        end_date=end_date,
        random_seed=seed,
    )
    saved_paths = generator.save_to_disk(raw_dir=raw_dir)

    print("[+] Successfully generated all datasets:")
    for name, path in saved_paths.items():
        print(f"    - {name:20s}: {path}")

    # Also save a sample subset in data/sample for quick inspection
    sample_dir = "data/sample"
    os.makedirs(sample_dir, exist_ok=True)
    import pandas as pd
    for table in ["products", "sales", "inventory"]:
        src_path = os.path.join(raw_dir, f"{table}.csv")
        sample_path = os.path.join(sample_dir, f"{table}_sample.csv")
        df = pd.read_csv(src_path, nrows=500)
        df.to_csv(sample_path, index=False)
    print(f"[+] Sample records saved to {sample_dir}")
    print("[*] Data generation complete.")


if __name__ == "__main__":
    main()
