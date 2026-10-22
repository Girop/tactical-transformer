from __future__ import annotations
from torch import nn
import torch
from dataclasses import dataclass
from tactics import SpecialTacticsTokens, CATALOG
from smt_embed import GIN_DIM


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    input_dimension: int
    model_dimension: int
    max_strat_len: int
    attention_heads: int
    layers: int
    feedforward: int

    @classmethod
    def default(cls):
        return cls(
            vocab_size=CATALOG.vocab_size(),
            input_dimension=GIN_DIM,
            model_dimension=768,
            max_strat_len=101,
            attention_heads=8,
            layers=12,
            feedforward=2048,
        )


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


class TacticTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.src_projection = nn.Linear(self.config.input_dimension, self.config.model_dimension)

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

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    @staticmethod
    def _embed(ids: torch.Tensor, tok_emb: nn.Embedding, pos_emb: nn.Embedding) -> torch.Tensor:
        pos = torch.arange(ids.size(1), device=ids.device)
        return tok_emb(ids) + pos_emb(pos)


    def _encode_src(self, smt: torch.Tensor) -> torch.Tensor:
        """The source is a single token, the GIN embedding: [B, input_dimension] -> [B, 1, model_dimension]."""
        return self.src_projection(smt).unsqueeze(1)


    def forward(self, smt: torch.Tensor, tgt_ids: torch.Tensor):
        src = self._encode_src(smt)
        tgt = self._embed(tgt_ids, self.tgt_embedding, self.tgt_pos_embedding)

        tgt_key_padding_mask = tgt_ids == SpecialTacticsTokens.PAD_ID.value

        x = self.transformer(
            src,
            tgt,
            tgt_mask=causal_mask(tgt_ids.size(1), tgt_ids.device),
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.out_projection(x)

    def _decode_step(self, ys: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Single decoder step, returns logits for the last position. [B, vocab_size]"""
        tgt = self._embed(ys, self.tgt_embedding, self.tgt_pos_embedding)
        out = self.transformer.decoder(
            tgt, memory,
            tgt_mask=causal_mask(ys.size(1), ys.device),
        )
        return self.out_projection(out[:, -1])

    @torch.no_grad()
    def generate(self, smt: torch.Tensor) -> torch.Tensor:
        """Greedy decode: one tactic id at a time, until every row has emitted EOS."""
        self.eval()
        batch = smt.size(0)
        memory = self.transformer.encoder(self._encode_src(smt))

        ys = torch.full((batch, 1), SpecialTacticsTokens.BOS_ID, dtype=torch.long, device=smt.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=smt.device)
        for _ in range(self.config.max_strat_len - 1):
            next_ids = self._decode_step(ys, memory).argmax(dim=-1)
            next_ids = next_ids.masked_fill(finished, SpecialTacticsTokens.PAD_ID)
            ys = torch.cat([ys, next_ids.unsqueeze(1)], dim=1)
            finished |= next_ids == SpecialTacticsTokens.EOS_ID
            if finished.all():
                break
        return ys


class DecoderOnlyTacticTransformer(nn.Module):
    """GIN prefix + causal decoder. The projected GIN embedding is prepended as
    position 0; tactic tokens follow at positions 1..seq_len."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.src_projection = nn.Linear(config.input_dimension, config.model_dimension)
        self.tgt_embedding = nn.Embedding(config.vocab_size, config.model_dimension)
        self.tgt_pos_embedding = nn.Embedding(config.max_strat_len, config.model_dimension)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.model_dimension,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.layers)
        self.out_projection = nn.Linear(config.model_dimension, config.vocab_size)

    @property
    def device(self):
        return next(self.decoder.parameters()).device

    def _prefix(self, smt: torch.Tensor) -> torch.Tensor:
        """[B, input_dim] -> [B, 1, model_dim]"""
        return self.src_projection(smt).unsqueeze(1)

    def _tactic_embed(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.size(1), device=ids.device)
        return self.tgt_embedding(ids) + self.tgt_pos_embedding(pos)

    def _forward_seq(self, seq: torch.Tensor, mask: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # TransformerDecoder without cross-attention: pass dummy memory of shape [B, 0, d]
        dummy_memory = seq.new_zeros(seq.size(0), 0, seq.size(2))
        return self.decoder(seq, dummy_memory, tgt_mask=mask, tgt_key_padding_mask=pad_mask)

    def forward(self, smt: torch.Tensor, tgt_ids: torch.Tensor):
        prefix = self._prefix(smt)                          # [B, 1, d]
        tactic = self._tactic_embed(tgt_ids)                # [B, S, d]
        seq = torch.cat([prefix, tactic], dim=1)            # [B, 1+S, d]

        total_len = seq.size(1)
        mask = causal_mask(total_len, seq.device)

        # Padding mask: prefix is never padded
        tactic_pad = tgt_ids == SpecialTacticsTokens.PAD_ID.value
        pad_mask = torch.cat([tactic_pad.new_zeros(tactic_pad.size(0), 1), tactic_pad], dim=1)

        out = self._forward_seq(seq, mask, pad_mask)
        return self.out_projection(out[:, 1:])              # logits for tactic positions only

    @torch.no_grad()
    def generate(self, smt: torch.Tensor) -> torch.Tensor:
        self.eval()
        batch = smt.size(0)
        prefix = self._prefix(smt)                          # [B, 1, d]

        bos = torch.full((batch, 1), SpecialTacticsTokens.BOS_ID, dtype=torch.long, device=smt.device)
        ys = bos
        finished = torch.zeros(batch, dtype=torch.bool, device=smt.device)

        for _ in range(self.config.max_strat_len - 1):
            tactic = self._tactic_embed(ys)
            seq = torch.cat([prefix, tactic], dim=1)
            mask = causal_mask(seq.size(1), seq.device)
            out = self._forward_seq(seq, mask)
            logits = self.out_projection(out[:, -1])        # last position

            next_ids = logits.argmax(dim=-1)
            next_ids = next_ids.masked_fill(finished, SpecialTacticsTokens.PAD_ID)
            ys = torch.cat([ys, next_ids.unsqueeze(1)], dim=1)
            finished |= next_ids == SpecialTacticsTokens.EOS_ID
            if finished.all():
                break
        return ys


def build_model(config: ModelConfig, architecture: str = "enc-dec") -> nn.Module:
    if architecture == "decoder-only":
        return DecoderOnlyTacticTransformer(config)
    if architecture == "enc-dec":
        return TacticTransformer(config)
    raise ValueError(f"Unknown architecture: {architecture}")
