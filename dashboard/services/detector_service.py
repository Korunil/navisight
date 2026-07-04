# dashboard/services/detector_service.py
import os
import time
import logging
import torch
from navisight.evaluation.ann_index import HierarchicalNavigableNetworkIndex
from navisight.evaluation.behavior_profile_engine import RollingBehaviorProfileEngine
from navisight.evaluation.inference_engine import ContextualDualChannelDetector

logger = logging.getLogger(__name__)

class DetectorService:
    def __init__(self):
        checkpoint_directory = "models/checkpoints/"
        global_stats_json_path = "configs/global_stats.json"
        index_directory = "models/state/index"
        
        best_checkpoint = os.path.join(checkpoint_directory, "best_model.pt")
        if not os.path.exists(best_checkpoint) and os.path.exists(checkpoint_directory):
            checkpoints = [os.path.join(checkpoint_directory, f) for f in os.listdir(checkpoint_directory) if f.endswith(".pt")]
            if checkpoints: best_checkpoint = checkpoints[-1]

        if not best_checkpoint or not os.path.exists(best_checkpoint):
            raise RuntimeError("Critical System Error: Best Model state checkpoint could not be resolved.")

        self.ann_index = HierarchicalNavigableNetworkIndex(index_dir=index_directory, dimension=128)
        
        # Operational verification check to confirm build_embeddings.py outputs are present
        if os.path.exists(index_directory):
            compiled_binaries = [f for f in os.listdir(index_directory) if f.endswith(".bin")]
            if compiled_binaries:
                logger.info(
                    "Successfully mapped index context to %d pre-compiled superclass HNSW binary sub-graphs.", 
                    len(compiled_binaries)
                )
            else:
                logger.warning(
                    "Zero compiled superclass .bin files discovered in %s. Check build_embeddings.py output.", 
                    index_directory
                )
        else:
            raise RuntimeError(f"Configured HNSW state directory does not exist: {index_directory}")

        self.profile_engine = RollingBehaviorProfileEngine(alpha=0.1)
        self.detector = ContextualDualChannelDetector(best_checkpoint, global_stats_json_path, self.ann_index, self.profile_engine)

    def evaluate(self, df_slice, vessel_id: int, context_meta: dict, mode: str = "production") -> dict:
        return self.detector.evaluate_live_sequence_anomaly_score(
            df_slice, vessel_id, context_meta, mode = mode
        )

    def query_neighbors(self, embedding, superclass_id: int, k: int = 3):
        return self.ann_index.query_behavior_neighborhood(embedding, superclass_id=superclass_id, k=k)