import os
import sqlite3 as sq
import sys
from pathlib import Path
from typing import Any

import pandas as pd

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))


class DataManager:
    """
    Manages data operations for production and inference workflows.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.db_path = config["data_manager"]["sqlite_db_path"]
        self.raw_data_table_name = config["data_manager"]["raw_data_table_name"]
        self.predictions_table_name = config["data_manager"]["predictions_table_name"]

    def init_raw_db_table(self) -> None:
        """
        Recreate the SQLite production database and load historical data.
        """
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        history_data_path = os.path.join(
            self.config["data_manager"]["history_data_folder"],
            self.config["data_manager"]["history_data_filename"],
        )
        df = pd.read_parquet(history_data_path)

        with sq.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")

            conn.execute(f"""
            CREATE TABLE {self.raw_data_table_name} (
                Timestamps TEXT PRIMARY KEY,
                WindSpeed REAL,
                WindDirAbs REAL,
                WindDirRel REAL,
                Power REAL,
                Pitch REAL,
                GenRPM REAL,
                RotorRPM REAL,
                EnvirTemp REAL,
                NacelTemp REAL,
                GearOilTemp REAL,
                GearBearTemp REAL,
                GenPh1Temp REAL,
                GenBearTemp REAL
            )
            """)

            df.to_sql(self.raw_data_table_name, conn, if_exists="append", index=False)

    def init_predictions_db_table(self) -> None:
        """
        Initialize the predictions table in the SQLite database.
        Creates the table and index if they don't exist.
        """
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        with sq.connect(self.db_path, timeout=30) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")

            # Create predictions table
            conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.predictions_table_name} (
                Timestamps TEXT NOT NULL PRIMARY KEY,
                predicted_power REAL NOT NULL
            )
            """)

            # Create index on Timestamps for faster queries
            conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_timestamps
            ON {self.predictions_table_name} (Timestamps)
            """)

    def get_last_n_points(self, n: int, table_name: str | None = None) -> pd.DataFrame:
        """
        Read the last N data points from a specified table in the SQLite database.

        Args:
            n: Number of points to retrieve
            table_name: Name of the table to read from. If None, defaults to raw_data_table_name.

        Returns:
            DataFrame containing the last N rows, ordered by Timestamps
        """
        if n <= 0:
            return pd.DataFrame()

        # Use provided table name or default to raw_data_table_name
        target_table = (
            table_name if table_name is not None else self.raw_data_table_name
        )

        with sq.connect(self.db_path, timeout=30) as conn:
            query = f"""
            SELECT * FROM {target_table}
            ORDER BY Timestamps DESC
            LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=[n])

            # Reverse to get chronological order (oldest to newest)
            df = df.iloc[::-1].reset_index(drop=True)

        return df

    def get_data_by_timestamp_range(
        self,
        start_timestamp: str | pd.Timestamp,
        end_timestamp: str | pd.Timestamp,
        table_name: str | None = None,
    ) -> pd.DataFrame:
        """
        Read data points within a timestamp range from a specified table.

        Args:
            start_timestamp: Start timestamp (inclusive). Can be string or pd.Timestamp.
            end_timestamp: End timestamp (inclusive). Can be string or pd.Timestamp.
            table_name: Name of the table to read from. If None, defaults to raw_data_table_name.

        Returns:
            DataFrame containing rows within the timestamp range, ordered by Timestamps
        """
        # Use provided table name or default to raw_data_table_name
        target_table = (
            table_name if table_name is not None else self.raw_data_table_name
        )

        # Convert timestamps to strings
        start_timestamp = str(start_timestamp)
        end_timestamp = str(end_timestamp)

        with sq.connect(self.db_path, timeout=30) as conn:
            query = f"""
            SELECT * FROM {target_table}
            WHERE Timestamps >= ? AND Timestamps <= ?
            ORDER BY Timestamps ASC
            """
            df = pd.read_sql_query(query, conn, params=[start_timestamp, end_timestamp])

        return df

    def insert_data_to_db(
        self,
        new_data: pd.DataFrame,
        table_name: str | None = None,
    ) -> None:
        """
        Write rows into a SQLite table in an idempotent way.

        Rules:
        - `Timestamps` column is REQUIRED for all writes.
        - Insert new rows.
        - If a row with the same `Timestamps` already exists, overwrite it (UPSERT).

        Requirements:
        - Target table must have `Timestamps` as PRIMARY KEY or UNIQUE.
          Example: `Timestamps TEXT PRIMARY KEY` or `UNIQUE(Timestamps)`.
        """
        # Nothing to write
        if new_data is None or new_data.empty:
            return

        # Enforce system invariant: every write is time-indexed
        if "Timestamps" not in new_data.columns:
            raise ValueError("Column `Timestamps` is required for all database writes.")

        # Resolve target table
        target_table = table_name or self.raw_data_table_name

        # Work on a copy to avoid mutating caller's DataFrame
        df = new_data.copy()

        # Normalize timestamps to prevent duplicates caused by formatting/precision differences
        df["Timestamps"] = pd.to_datetime(df["Timestamps"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Extract column names from the DataFrame
        cols = list(df.columns)

        # Build a comma-separated list of quoted column names for SQL
        # Example: "Timestamps","y_pred","model_version"
        col_list = ", ".join(f'"{c}"' for c in cols)

        # Create positional placeholders (?, ?, ?) for parameterized INSERT
        placeholders = ", ".join(["?"] * len(cols))

        # All columns except the timestamp key will be updated on conflict
        update_cols = [c for c in cols if c != "Timestamps"]

        # If the table only contains the key column, there is nothing to update
        if not update_cols:
            return

        # Build SET clause for UPSERT using SQLite's `excluded` alias
        # Example: "y_pred"=excluded."y_pred","model_version"=excluded."model_version"
        set_clause = ", ".join(f'"{c}" = excluded."{c}"' for c in update_cols)

        # Final UPSERT statement:
        # - INSERT new rows
        # - UPDATE existing rows when the same timestamp already exists
        sql = f"""
        INSERT INTO "{target_table}" ({col_list})
        VALUES ({placeholders})
        ON CONFLICT("Timestamps") DO UPDATE SET
            {set_clause};
        """

        # Open database connection and execute batch UPSERT
        with sq.connect(self.db_path, timeout=30) as conn:
            # Enable Write-Ahead Logging for safer concurrent access
            conn.execute("PRAGMA journal_mode=WAL;")

            # Batch execution is efficient and keeps the operation atomic
            # Use values.tolist() to ensure correct column order and tuple format
            conn.executemany(sql, df[cols].values.tolist())
            conn.commit()
