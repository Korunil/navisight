# navisight/evaluation/behavior_profile_engine.py
from collections import deque
import numpy as np

class RollingBehaviorProfileEngine:
    """Maintains an exponential moving average prototype vector per tracking vessel."""
    def __init__(self, alpha: float = 0.1, base_calibration_percentile: float = 95.0):
        self.alpha = alpha
        self.percentile = base_calibration_percentile
        self.vessel_prototype_matrix = {}
        
        # FIXED: Replaced inefficient Python lists with rapid O(1) double-ended queues
        self.calibration_buffer = deque(maxlen=10000)
        self.dynamic_threshold = 0.45 

    def update_profile_and_score_drift(self, vessel_id_int: int, current_embedding: np.ndarray) -> dict:
        norm_factor = np.linalg.norm(current_embedding)
        emb_clean = current_embedding / (norm_factor if norm_factor > 1e-8 else 1.0)
        
        if vessel_id_int not in self.vessel_prototype_matrix:
            self.vessel_prototype_matrix[vessel_id_int] = emb_clean.copy()
            return {"cosine_drift_score": 0.0, "is_manifold_outlier": False}
            
        prototype = self.vessel_prototype_matrix[vessel_id_int]
        cosine_distance = 1.0 - float(np.dot(prototype, emb_clean))
        
        self.calibration_buffer.append(cosine_distance)
        if len(self.calibration_buffer) >= 1000:
            self.dynamic_threshold = float(np.percentile(list(self.calibration_buffer), self.percentile))
        
        updated_prototype = (self.alpha * emb_clean) + ((1.0 - self.alpha) * prototype)
        self.vessel_prototype_matrix[vessel_id_int] = updated_prototype / max(np.linalg.norm(updated_prototype), 1e-8)
        
        return {
            "cosine_drift_score": cosine_distance,
            "is_manifold_outlier": cosine_distance > self.dynamic_threshold
        }