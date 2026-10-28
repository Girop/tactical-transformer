from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from tactics import SpecialTacticsTokens
from utils import ModelConfig


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


class TacticGenerator(ABC, nn.Module):
    config: ModelConfig

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...

    @abstractmethod
    def _encode(self, smt: torch.Tensor) -> torch.Tensor:
        """Build the initial `memory` passed to `_decode_step` at every generation step."""
        ...

    @abstractmethod
    def _decode_step(self, ys: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Single decoder step, returns logits for the last position. [B, vocab_size]"""
        ...

    @torch.no_grad()
    def generate(self, smt: torch.Tensor) -> torch.Tensor:
        """Greedy decode: one tactic id at a time, until every row has emitted EOS."""
        self.eval()
        batch = smt.size(0)
        memory = self._encode(smt)

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


class TacticTransformer(TacticGenerator):
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

    def _encode(self, smt: torch.Tensor) -> torch.Tensor:
        return self.transformer.encoder(self._encode_src(smt))

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


class DecoderOnlyTacticTransformer(TacticGenerator):
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

    def _encode(self, smt: torch.Tensor) -> torch.Tensor:
        return self._prefix(smt)

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
        return self.out_projection(out[:, 1:])

    def _decode_step(self, ys: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Single decoder step, returns logits for the last position. [B, vocab_size]"""
        tactic = self._tactic_embed(ys)
        seq = torch.cat([memory, tactic], dim=1)
        mask = causal_mask(seq.size(1), seq.device)
        out = self._forward_seq(seq, mask)
        return self.out_projection(out[:, -1])


class LoRATacticGenerator(TacticGenerator):

    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    LORA_MODULES_TO_SAVE = ["embed_tokens", "lm_head"]

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        hf_config = AutoConfig.from_pretrained(config.pretrained_model_name)
        hf_config.tie_word_embeddings = False
        # Match the new (float32) embedding/head/projection layers below; the checkpoint
        # may default to bf16, and this codebase doesn't use mixed-precision training.
        llm = AutoModelForCausalLM.from_pretrained(config.pretrained_model_name, config=hf_config, dtype=torch.float32)
        self.model_dimension = llm.config.hidden_size

        llm.set_input_embeddings(nn.Embedding(config.vocab_size, self.model_dimension))
        llm.set_output_embeddings(nn.Linear(self.model_dimension, config.vocab_size, bias=False))

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.LORA_R,
            lora_alpha=self.LORA_ALPHA,
            lora_dropout=self.LORA_DROPOUT,
            target_modules=self.LORA_TARGET_MODULES,
            modules_to_save=self.LORA_MODULES_TO_SAVE,
        )
        self.llm = get_peft_model(llm, lora_config)

        self.src_projection = nn.Linear(config.input_dimension, self.model_dimension)

    @property
    def device(self) -> torch.device:
        return next(self.llm.parameters()).device

    def _tactic_embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(ids)

    def _prefix(self, smt: torch.Tensor) -> torch.Tensor:
        """[B, input_dim] -> [B, 1, model_dim]"""
        return self.src_projection(smt).unsqueeze(1)

    def _encode(self, smt: torch.Tensor) -> torch.Tensor:
        return self._prefix(smt)

    def forward(self, smt: torch.Tensor, tgt_ids: torch.Tensor):
        prefix = self._prefix(smt)                          # [B, 1, d]
        tactic = self._tactic_embed(tgt_ids)                # [B, S, d]
        seq = torch.cat([prefix, tactic], dim=1)            # [B, 1+S, d]

        # Padding mask: prefix is never padded. Causal masking is applied internally by the backbone.
        attended = tgt_ids != SpecialTacticsTokens.PAD_ID.value
        attention_mask = torch.cat([attended.new_ones(attended.size(0), 1), attended], dim=1)

        out = self.llm(inputs_embeds=seq, attention_mask=attention_mask).logits
        return out[:, 1:]

    def _decode_step(self, ys: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Single decoder step, returns logits for the last position. [B, vocab_size]"""
        tactic = self._tactic_embed(ys)
        seq = torch.cat([memory, tactic], dim=1)
        out = self.llm(inputs_embeds=seq).logits
        return out[:, -1]


def build_model(config: ModelConfig) -> nn.Module:
    if config.architecture == "dec":
        return DecoderOnlyTacticTransformer(config)
    if config.architecture == "enc-dec":
        return TacticTransformer(config)
    if config.architecture == "llm-lora":
        return LoRATacticGenerator(config)
    raise ValueError(f"Unknown architecture: {config.architecture}")
