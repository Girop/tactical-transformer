from torch import nn
import torch
from z3alpha.tactics.catalog import  SOLVER_CATALOG, PREPROCESS_CATALOG
from enum import Enum
from dataclasses import dataclass
from smt_graph import NUM_OPS


SRC_PAD_ID = NUM_OPS

class SpecialTacticsTokens(Enum):
    __LAST_STRAT_ID = max([*SOLVER_CATALOG.keys(), *PREPROCESS_CATALOG.keys()])

    PAD_ID = __LAST_STRAT_ID + 1
    UNK_ID = __LAST_STRAT_ID + 2
    BOS_ID = __LAST_STRAT_ID + 3
    EOS_ID = __LAST_STRAT_ID + 4

    @classmethod
    def vocab_size(cls):
        return cls.EOS_ID.value


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
        self.causal_mask = torch.triu(torch.ones(self.config.max_strat_len, self.config.max_strat_len, device=device, dtype=torch.bool), diagonal=1)


    @staticmethod
    def _embed(ids: torch.Tensor, tok_emb: nn.Embedding, pos_emb: nn.Embedding) -> torch.Tensor:
        pos = torch.arange(ids.size(1), device=ids.device)
        return tok_emb(ids) + pos_emb(pos)


    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor):
        src = self._embed(src_ids, self.src_embedding, self.src_pos_embedding)
        tgt = self._embed(tgt_ids, self.tgt_embedding, self.tgt_pos_embedding)

        src_key_padding_mask = src_ids == SRC_PAD_ID
        tgt_key_padding_mask = tgt_ids == SpecialTacticsTokens.PAD_ID.value

        x = self.transformer(
            src,
            tgt,
            tgt_mask=self.causal_mask[:tgt_ids.size(1), :tgt_ids.size(1)],
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

        ys = torch.full((batch, 1), SpecialTacticsTokens.BOS_ID.value, dtype=torch.long, device=smt.device)
        for _ in range(self.config.max_strat_len - 1):
            tgt = self._embed(ys, self.tgt_embedding, self.tgt_pos_embedding)
            out = self.transformer.decoder(
                tgt, memory,
                tgt_mask=self.causal_mask[:ys.size(1), :ys.size(1)],
                memory_key_padding_mask=src_key_padding_mask,
            )
            next_ids = self.out_projection(out[:, -1]).argmax(dim=-1)
            ys = torch.cat([ys, next_ids.unsqueeze(1)], dim=1)
            if bool((next_ids == SpecialTacticsTokens.EOS_ID.value).all()):
                break
        return ys

