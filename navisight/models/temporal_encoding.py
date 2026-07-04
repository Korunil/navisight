# navisight/models/temporal_encoding.py
import logging
import torch
import torch.nn as nn
import math

class FoundationSpatiotemporalPositionEncoder(nn.Module):
    def __init__(self, d_model: int, max_window_size: int = 200, max_time_scale: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.absolute_position_embed = nn.Embedding(max_window_size, d_model)
        
        self.delta_time_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.cum_time_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(max_time_scale) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, log_time_diffs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = log_time_diffs.size()
        device = log_time_diffs.device
        
        delta_expanded = log_time_diffs.unsqueeze(-1)
        pe_delta = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        pe_delta[:, :, 0::2] = torch.sin(delta_expanded * self.div_term)
        pe_delta[:, :, 1::2] = torch.cos(delta_expanded * self.div_term)
        h_delta = self.delta_time_mlp(pe_delta)
        
        cum_time = torch.cumsum(log_time_diffs, dim=1)
        cum_expanded = cum_time.unsqueeze(-1)
        pe_cum = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        pe_cum[:, :, 0::2] = torch.sin(cum_expanded * self.div_term)
        pe_cum[:, :, 1::2] = torch.cos(cum_expanded * self.div_term)
        h_cum = self.cum_time_mlp(pe_cum)
        
        seq_positions = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # ── FORENSIC LAYER CAPACITY GUARD ──
        # Directly reads the hardware row allocation count from the neural weights matrix
        max_allocated_rows = self.absolute_position_embed.num_embeddings
        max_requested_position = seq_positions.max().item()
        
        # Log the exact structural dimensions on the very first forward pass step
        if not hasattr(self, "_capacity_logged"):
            logging.info("==========================================================================")
            logging.info(
                f"🔮 TEMPORAL CAPACITIES | Sequence Length: {seq_positions.size(1)} | "
                f"Requested Max Position: {max_requested_position} | "
                f"Embedding Weights Rows: {max_allocated_rows}"
            )
            logging.info("==========================================================================")
            self._capacity_logged = True

        # FAIL-FAST: Halt processing on the CPU before a CUDA boundary crash can occur
        if max_requested_position >= max_allocated_rows:
            raise RuntimeError(
                f"\n🚨 POSITION EMBEDDING OVERFLOW DETECTED 🚨\n"
                f"The model's absolute_position_embed layer has a hard capacity ceiling of exactly [{max_allocated_rows}] rows,\n"
                f"but the incoming tracking stream requested a position mapping index up to [{max_requested_position}].\n"
                f"This will cause out-of-bounds CUDA parallel kernel violations!\n"
                f"Action Required: Align your sequence context lengths with your training boundaries."
            )
        
        h_pos = self.absolute_position_embed(seq_positions)
        
        cls_placeholder = torch.zeros(batch_size, 1, self.d_model, device=device)
        trajectory_signals = h_pos + h_delta + h_cum
        
        return torch.cat([cls_placeholder, trajectory_signals], dim=1)