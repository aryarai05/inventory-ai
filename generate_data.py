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
from src.data.uci_processor import process_uci_dataset


def main():
    print("[*] Generating supply chain dataset from real UCI Online Retail data (541k records)...")
    try:
        process_uci_dataset(num_skus=60)
        print("[+] Successfully generated real-world dataset from UCI Online Retail!")
    except Exception as e:
        print(f"[!] Real dataset pipeline exception ({e}), falling back to parametric generator...")
        generator = SyntheticInventoryDataGenerator(
            start_date="2023-01-01",
            end_date="2025-12-31",
            random_seed=42,
        )
        generator.save_to_disk(raw_dir="data/raw")
        print("[+] Parametric dataset generation complete.")


if __name__ == "__main__":
    main()

