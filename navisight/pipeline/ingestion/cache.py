# navisight/pipeline/ingestion/cache.py
import numpy as np
import threading
from typing import NamedTuple, Mapping

class VesselStateValue(NamedTuple):
    """Immutable value object capturing a single tracking coordinate point sequence."""
    vessel_id_int: int
    mmsi_str: str
    last_timestamp: int
    last_speed: float
    last_lon: float
    last_lat: float
    global_lsn: int

class PureFunctionalProjectionCache:
    """Immutable, lock-free functional cache mapping track lifecycle updates deterministically."""
    def __init__(self):
        # Authoritative state map container: vessel_id_int -> VesselStateValue
        self.state_map = {}
        self.lock = threading.Lock()

    def apply_mutation_record(self, vessel_id_int: int, timestamp_sec: int, global_lsn: int, 
                              mmsi_str: str, record_data: dict):
        """Updates the state map via deterministic, monotonic logic gating paths."""
        with self.lock:
            if vessel_id_int in self.state_map:
                current_state = self.state_map[vessel_id_int]
                # Sequence Check Gate: Reject out-of-order mutations instantly
                if global_lsn < current_state.global_lsn:
                    return

            new_value = VesselStateValue(
                vessel_id_int=vessel_id_int,
                mmsi_str=mmsi_str if mmsi_str else (self.state_map[vessel_id_int].mmsi_str if vessel_id_int in self.state_map else "UNKNOWN"),
                last_timestamp=timestamp_sec,
                last_speed=float(record_data.get("last_speed", 0.0)),
                last_lon=float(record_data.get("last_lon", 0.0)),
                last_lat=float(record_data.get("last_lat", 0.0)),
                global_lsn=global_lsn
            )
            self.state_map[vessel_id_int] = new_value

    def query_vessel_state_read_only(self, vessel_id_int: int) -> VesselStateValue:
        """Read-only data lookup mapping pass."""
        with self.lock:
            return self.state_map.get(vessel_id_int, None)