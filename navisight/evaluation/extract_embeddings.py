# navisight/evaluation/extract_embeddings.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from navisight.pipeline.sequence_builder import ContinuityPreservingAISDataset
from navisight.models.transformer_encoder import MaritimeMAE
from navisight.evaluation.embedding_store import AppendOnlyEmbeddingStore

class ZeroCopyExtractionPipeline:
    """Extracts hidden CLS tokens and serializes them to float32 Parquet shards."""
    def __init__(self, checkpoint_path: str, embedding_destination_dir: str, device_type: str = "cuda"):
        self.device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            
        self.model = MaritimeMAE(d_model=128, n_heads=8, n_layers=4).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.store = AppendOnlyEmbeddingStore(base_dir=embedding_destination_dir)

    def extract_and_serialize_latent_manifold(self, data_source_partitions_dir: str, config_json: str, batch_size: int = 512):
        dataset = ContinuityPreservingAISDataset(partitioned_root_dir=data_source_partitions_dir, manifest_json_path=config_json)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

        global_embedding_id = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="⚡ Extracting Latent Manifolds"):
                features = batch['features'].to(self.device)
                attn_mask = batch['attention_mask'].to(self.device)
                
                _, cls_embeddings, _, _ = self.model(features, attn_mask, mask_ratio=0.0)
                np_cls_vectors = cls_embeddings.cpu().numpy()
                batch_rows_count = len(np_cls_vectors)
                
                embedding_ids = np.arange(global_embedding_id, global_embedding_id + batch_rows_count, dtype=np.int64)
                global_embedding_id += batch_rows_count
                
                # FIXED: Eliminated the hazardous string-to-int type-cast statements
                metadata_payload = {
                    "embedding_id": embedding_ids.tolist(),
                    "vessel_id_int": batch["vessel_id_int"].numpy().astype(np.int64).tolist(),
                    "timestamp_sec": batch["timestamp_sec"].numpy().astype(np.int64).tolist(),
                    "trip_id": [str(t) for t in batch["trip_id"]],
                    "superclass_id": batch["superclass_id"].numpy().astype(np.int16).tolist(),
                    "reliability": batch["reliability"][:, -1].numpy().astype(np.float32).tolist()
                }

                self.store.append_latent_vectors(metadata_payload, np_cls_vectors)
                
        # Ensure remaining buffered rows are flushed to disk before the tracking step ends
        self.store.close()