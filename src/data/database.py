"""
Database Management Module
Handles SQLite schema creation, data loading, and execution of analytical SQL queries.
"""

import os
import sqlite3
import pandas as pd
from typing import Dict, Any, Optional


class DatabaseManager:
    def __init__(self, db_path: str = "data/inventory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def initialize_schema(self, schema_file: str = "sql/schema.sql") -> None:
        """Executes the DDL statements in schema.sql to initialize all tables."""
        if not os.path.exists(schema_file):
            raise FileNotFoundError(f"Schema file not found at: {schema_file}")

        with open(schema_file, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self.get_connection() as conn:
            conn.executescript(schema_sql)

    def load_table(self, table_name: str, df: pd.DataFrame, if_exists: str = "append") -> None:
        """Loads a pandas DataFrame into a specified SQLite table."""
        with self.get_connection() as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)

    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Executes a SQL SELECT query and returns the results as a pandas DataFrame."""
        with self.get_connection() as conn:
            if params:
                return pd.read_sql_query(query, conn, params=params)
            return pd.read_sql_query(query, conn)

    def execute_sql_file(self, sql_file: str) -> Dict[str, pd.DataFrame]:
        """
        Parses and executes queries separated by semicolons in a SQL script file,
        returning a dictionary of results mapped by comment header or query index.
        """
        if not os.path.exists(sql_file):
            raise FileNotFoundError(f"SQL file not found at: {sql_file}")

        with open(sql_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Split on comments or double newlines to isolate distinct queries
        queries = [q.strip() for q in content.split(";") if q.strip() and not q.strip().startswith("-- ==")]
        results = {}

        with self.get_connection() as conn:
            for idx, q in enumerate(queries):
                # Clean up leading comments for identification
                lines = [line.strip() for line in q.splitlines() if line.strip()]
                title = f"Query_{idx+1}"
                for line in lines:
                    if line.startswith("--"):
                        clean_title = line.lstrip("-").strip()
                        if clean_title:
                            title = clean_title
                            break
                try:
                    df = pd.read_sql_query(q, conn)
                    results[title] = df
                except Exception as e:
                    # Ignore comment-only blocks or errors in non-SELECT statements
                    pass

        return results
