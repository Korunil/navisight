# navisight/pipeline/ingestion/database.py
import sqlite3
import threading
import time
import logging

logger = logging.getLogger(__name__)

class LineageDatabaseBroker:
    """Thread-isolated SQLite state engine acting as a strict structured projection view."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db_lock = threading.Lock()
        self._init_db()
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA wal_autocheckpoint=10000;")
        self.conn.execute("PRAGMA busy_timeout=5000;")

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vessel_metadata_lineage (
                    vessel_id_int INTEGER PRIMARY KEY,
                    last_timestamp INTEGER,
                    last_speed REAL,
                    last_lon REAL,
                    last_lat REAL,
                    last_is_static INTEGER,
                    active_static_duration INTEGER,
                    trip_id_offset INTEGER,
                    last_seen_epoch INTEGER,
                    global_lsn INTEGER DEFAULT -1
                );
            """)
            conn.commit()

    def apply_log_records(self, records: list):
        """Merges log arrays into the structured projection view, gated by logical sequence metrics."""
        if not records:
            return
            
        with self.db_lock:
            upsert_query = """
                INSERT INTO vessel_metadata_lineage (
                    vessel_id_int, last_timestamp, last_speed, last_lon, last_lat,
                    last_is_static, active_static_duration, trip_id_offset, last_seen_epoch,
                    global_lsn
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(vessel_id_int) DO UPDATE SET
                    last_timestamp = excluded.last_timestamp,
                    last_speed = excluded.last_speed,
                    last_lon = excluded.last_lon,
                    last_lat = excluded.last_lat,
                    last_is_static = excluded.last_is_static,
                    active_static_duration = excluded.active_static_duration,
                    trip_id_offset = excluded.trip_id_offset,
                    last_seen_epoch = excluded.last_seen_epoch,
                    global_lsn = excluded.global_lsn
                WHERE excluded.global_lsn >= vessel_metadata_lineage.global_lsn;
            """
            try:
                if not self.conn.in_transaction:
                    self.conn.execute("BEGIN;")
                self.conn.executemany(upsert_query, records)
                self.conn.commit()
            except Exception as e:
                if self.conn.in_transaction:
                    self.conn.rollback()
                logger.error(f"Failed to apply batch records to database projection: {e}")
                raise e

    def fetch_vessel_history(self, vessel_id_int: int) -> tuple:
        """Fetches historical tracking markers from the structured relational projection index."""
        with self.db_lock:
            cursor = self.conn.execute(
                """
                SELECT last_timestamp, last_speed, last_lon, last_lat, last_is_static,
                       active_static_duration, trip_id_offset, last_seen_epoch, global_lsn
                FROM vessel_metadata_lineage WHERE vessel_id_int = ?;
                """, (vessel_id_int,)
            )
            try:
                return cursor.fetchone()
            finally:
                cursor.close()

    def close(self):
        with self.db_lock:
            try:
                if self.conn.in_transaction:
                    self.conn.commit()
                self.conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
            finally:
                self.conn.close()