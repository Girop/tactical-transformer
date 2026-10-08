from torch import nn
import torch
from dataclasses import dataclass
from smt_graph import NUM_OPS
from tactics import SpecialTacticsTokens

SRC_PAD_ID = NUM_OPS


@dataclass(frozen=True)
class ModelConfig:
    graph_tensor_size: int
    max_graph_size: int
    vocab_size: int
    model_dimension: int
    max_strat_len: int
    attention_heads: int
    layers: int
    feedforward: int


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


class TacticTransformer(nn.Module):
    def __init__(self, config: ModelConfig, device) -> None:
        super().__init__()
        self.config = config
        self.device = device

        self.src_embedding = nn.Embedding(self.config.graph_tensor_size, self.config.model_dimension)
        self.src_pos_embedding = nn.Embedding(self.config.max_graph_size, self.config.model_dimension)

        self.tgt_embedding = nn.Embedding(self.config.vocab_size, self.config.model_dimension)
        self.tgt_pos_embedding = nn.Embedding(self.config.max_strat_len, self.config.model_dimension)

        self.transformer = nn.Transformer(
            d_model=self.config.model_dimension,
            nhead=self.config.attention_heads,
            num_encoder_layers=self.config.layers,
            num_decoder_layers=self.config.layers,
            dim_feedforward=self.config.feedforward,
            batch_first=True
        )

        self.out_projection = nn.Linear(self.config.model_dimension, self.config.vocab_size)


    @staticmethod
    def _embed(ids: torch.Tensor, tok_emb: nn.Embedding, pos_emb: nn.Embedding) -> torch.Tensor:
        pos = torch.arange(ids.size(1), device=ids.device)
        return tok_emb(ids) + pos_emb(pos)


    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor):
        src = self._embed(src_ids, self.src_embedding, self.src_pos_embedding)
        tgt = self._embed(tgt_ids, self.tgt_embedding, self.tgt_pos_embedding)

        src_key_padding_mask = src_ids == SRC_PAD_ID
        tgt_key_padding_mask = tgt_ids == SpecialTacticsTokens.PAD_ID

        x = self.transformer(
            src,
            tgt,
            tgt_mask=causal_mask(tgt_ids.size(1), tgt_ids.device),
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.out_projection(x)

    @torch.no_grad()
    def generate(self, smt: torch.Tensor) -> torch.Tensor:
        """Greedy decode: one tactic id at a time, until every row has emitted EOS."""
        self.eval()
        batch = smt.size(0)

        src = self._embed(smt, self.src_embedding, self.src_pos_embedding)
        src_key_padding_mask = smt == SRC_PAD_ID
        memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)

        ys = torch.full((batch, 1), SpecialTacticsTokens.BOS_ID, dtype=torch.long, device=smt.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=smt.device)
        for _ in range(self.config.max_strat_len - 1):
            tgt = self._embed(ys, self.tgt_embedding, self.tgt_pos_embedding)
            out = self.transformer.decoder(
                tgt, memory,
                tgt_mask=causal_mask(ys.size(1), ys.device),
                memory_key_padding_mask=src_key_padding_mask,
            )
            next_ids = self.out_projection(out[:, -1]).argmax(dim=-1)
            # Rows that already emitted EOS only get padding, so nothing after EOS reaches the strategy.
            next_ids = next_ids.masked_fill(finished, SpecialTacticsTokens.PAD_ID)
            ys = torch.cat([ys, next_ids.unsqueeze(1)], dim=1)
            finished |= next_ids == SpecialTacticsTokens.EOS_ID
            if finished.all():
                break
        return ys

