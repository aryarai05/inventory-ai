"""
Scalable PySpark Feature Preprocessing Pipeline
Demonstrates enterprise distributed processing of massive multi-million transaction tables
using Apache Spark DataFrame APIs and Window functions.
Note: Optional module for Spark/Databricks environments.
"""

import sys
from typing import Optional

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


def build_spark_session(app_name: str = "InventoryDemandFeaturePipeline") -> Optional["SparkSession"]:
    """Initializes a local or cluster PySpark session."""
    if not SPARK_AVAILABLE:
        print("[!] PySpark is not installed in the local environment. Skipping Spark execution.")
        return None
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def run_spark_feature_pipeline(
    sales_csv_path: str = "data/raw/sales.csv",
    output_parquet_path: str = "data/processed/spark_features.parquet",
) -> None:
    """
    Executes distributed time-series feature engineering with PySpark.
    """
    spark = build_spark_session()
    if spark is None:
        return

    print(f"[*] Reading raw sales data via Spark from: {sales_csv_path}")
    sales_df = spark.read.option("header", "true").option("inferSchema", "true").csv(sales_csv_path)

    # Daily aggregation per product across regions
    daily_sales = (
        sales_df
        .groupBy("date", "product_id")
        .agg(
            F.sum("sales_quantity").alias("sales_quantity"),
            F.sum("unmet_demand").alias("unmet_demand"),
            F.sum("revenue").alias("revenue"),
            F.max("promotion_flag").alias("promotion_flag")
        )
    )

    # Define time-series partitioning window per product ordered by date
    window_spec = Window.partitionBy("product_id").orderBy("date")

    # 1. Distributed Lag Features
    print("[*] Generating distributed time-series lags...")
    df_lags = (
        daily_sales
        .withColumn("lag_1", F.lag("sales_quantity", 1).over(window_spec))
        .withColumn("lag_7", F.lag("sales_quantity", 7).over(window_spec))
        .withColumn("lag_14", F.lag("sales_quantity", 14).over(window_spec))
        .withColumn("lag_28", F.lag("sales_quantity", 28).over(window_spec))
    )

    # 2. Shifted Rolling Averages over past 7 and 14 days (rowsBetween(-7, -1))
    # Excludes current row to guarantee zero lookahead leakage!
    rolling_7_spec = Window.partitionBy("product_id").orderBy("date").rowsBetween(-7, -1)
    rolling_14_spec = Window.partitionBy("product_id").orderBy("date").rowsBetween(-14, -1)

    print("[*] Computing non-leaking distributed rolling statistics...")
    df_features = (
        df_lags
        .withColumn("rolling_mean_7", F.avg("sales_quantity").over(rolling_7_spec))
        .withColumn("rolling_mean_14", F.avg("sales_quantity").over(rolling_14_spec))
        .withColumn("rolling_std_7", F.stddev("sales_quantity").over(rolling_7_spec))
    )

    print(f"[*] Writing distributed parquet features to: {output_parquet_path}")
    df_features.write.mode("overwrite").parquet(output_parquet_path)
    print("[+] PySpark pipeline completed successfully.")


if __name__ == "__main__":
    run_spark_feature_pipeline()
