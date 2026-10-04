"""
Creates the 5 professional Jupyter Notebooks for the project.
"""

import json
import os


def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.12.7"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in src.strip().split("\n")],
    }


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src.strip().split("\n")],
    }


def main():
    nb_dir = "notebooks"
    os.makedirs(nb_dir, exist_ok=True)

    # 01_data_exploration.ipynb
    nb1 = make_nb([
        md("# 01. Data Exploration & Raw Data Ingestion\n## AI-Powered Inventory Demand Forecasting & Optimization System\nThis notebook inspects the raw transactional datasets, catalogs, and supplier agreements."),
        code("import pandas as pd\nimport numpy as np\nimport os\n\nraw_dir = '../data/raw'\nproducts = pd.read_csv(os.path.join(raw_dir, 'products.csv'))\nsales = pd.read_csv(os.path.join(raw_dir, 'sales.csv'))\ninventory = pd.read_csv(os.path.join(raw_dir, 'inventory.csv'))\nprint('Products:', products.shape)\nprint('Sales Transactions:', sales.shape)\nprint('Inventory Snapshots:', inventory.shape)"),
        code("products.head(5)"),
        code("sales.info()"),
        code("inventory.describe()")
    ])
    with open(os.path.join(nb_dir, "01_data_exploration.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=2)

    # 02_eda.ipynb
    nb2 = make_nb([
        md("# 02. Exploratory Data Analysis & Supply Chain Diagnostics\nIn this notebook, we analyze demand seasonality, promo lifts, regional allocations, and stockout history."),
        code("import pandas as pd\nimport sys\nsys.path.append('..')\nfrom src.visualization.charts import *\n\nsales = pd.read_csv('../data/raw/sales.csv')\nproducts = pd.read_csv('../data/raw/products.csv')\nsales = sales.merge(products[['product_id', 'product_name', 'category']], on='product_id', how='left')\ninventory = pd.read_csv('../data/raw/inventory.csv')"),
        code("fig_trend = plot_overall_sales_trend(sales)\nfig_trend.show()"),
        code("fig_cat = plot_category_demand(sales)\nfig_cat.show()"),
        code("fig_heat = plot_seasonality_patterns(sales)\nfig_heat.show()"),
        code("fig_stockout = plot_stockout_frequency(inventory)\nfig_stockout.show()")
    ])
    with open(os.path.join(nb_dir, "02_eda.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=2)

    # 03_feature_engineering.ipynb
    nb3 = make_nb([
        md("# 03. Feature Engineering & Lookahead Leakage Prevention\nConstructs lags (1, 7, 14, 28), shifted rolling statistics, cyclical calendar features, and economic signals."),
        code("import pandas as pd\nimport sys\nsys.path.append('..')\nfrom src.features.engineer import FeatureEngineer\n\nmaster_df = pd.read_csv('../data/processed/featured_dataset.csv')\nprint('Feature Matrix Shape:', master_df.shape)\nprint('Engineered Columns:', [c for c in master_df.columns if c.startswith('lag_') or c.startswith('rolling_')])"),
        code("master_df[['date', 'product_id', 'sales_quantity', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7']].head(10)")
    ])
    with open(os.path.join(nb_dir, "03_feature_engineering.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb3, f, indent=2)

    # 04_model_training.ipynb
    nb4 = make_nb([
        md("# 04. Model Training, LightGBM Tuning & MLflow Tracking\nTrains baseline heuristics, Ridge regression, Random Forest, and LightGBM with chronological validation."),
        code("import pandas as pd\nimport json\nimport sys\nsys.path.append('..')\n\nwith open('../models/model_comparison.json', 'r') as f:\n    leaderboard = json.load(f)\npd.DataFrame(leaderboard)")
    ])
    with open(os.path.join(nb_dir, "04_model_training.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb4, f, indent=2)

    # 05_model_evaluation.ipynb
    nb5 = make_nb([
        md("# 05. Model Evaluation, Residual Analysis & Out-of-Sample Performance\nDetailed inspection of prediction errors, residual distributions, and feature importance."),
        code("import pandas as pd\nimport plotly.express as px\n\ntest_preds = pd.read_csv('../data/processed/test_predictions.csv')\nfeat_imp = pd.read_csv('../models/feature_importance.csv')\nprint('Top 10 Features:')\nfeat_imp.head(10)"),
        code("fig_err = px.histogram(test_preds, x='forecast_error', nbins=50, title='Out-of-Sample Residual Distribution')\nfig_err.show()")
    ])
    with open(os.path.join(nb_dir, "05_model_evaluation.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb5, f, indent=2)

    print("All 5 Jupyter notebooks generated successfully.")


if __name__ == "__main__":
    main()
