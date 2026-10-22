import logging

from pathlib import Path
from transformer import TacticTransformer
from math import inf
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm

from utils import ExperimentConfig
from tactics import SpecialTacticsTokens

log = logging.getLogger(__name__)


class Par2Loss(nn.Module):
    def __init__(self, timeout: float, pad_id: int, label_smoothing: float = 0.0):
        super().__init__()
        self.timeout = timeout
        self.pad_id = pad_id
        self.ce = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='none', label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, benchmarks: list) -> torch.Tensor:
        tokens = (targets != self.pad_id).sum(dim=1).clamp(min=1)
        per_sample = self.ce(logits.transpose(1, 2), targets).sum(dim=1) / tokens
        weights = torch.tensor([max(0.0, 1.0 - b.time_s / self.timeout) if b.solved else 0.0 for b in benchmarks], device=logits.device)
        return (per_sample * weights).mean()


class CELoss(nn.Module):
    def __init__(self, pad_id: int, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, benchmarks: list) -> torch.Tensor:
        return self.ce(logits.transpose(1, 2), targets)


@torch.no_grad()
def mean_loss(model: TacticTransformer, loader: DataLoader) -> float:
    model.eval()
    total = 0.0
    count = 0
    for smt, strat, _ in loader:
        smt, strat = smt.to(model.device), strat.to(model.device)
        logits = model(smt, strat[:, :-1])
        bs = smt.size(0)
        total += F.cross_entropy(logits.transpose(1, 2), strat[:, 1:], ignore_index=SpecialTacticsTokens.PAD_ID).item() * bs
        count += bs
    return total / count


class Trainer:

    def __init__(
        self,
        model: TacticTransformer,
        train: DataLoader,
        validation: DataLoader,
        config: ExperimentConfig,
        timeout: float = 10.0,
        model_storage: Path = Path("models")
    ) -> None:
        self.model = model
        self.train = train
        self.validation = validation
        self.config = config
        self.timeout = timeout
        self.model_storage = model_storage
        model_storage.mkdir(exist_ok=True)


    def __criterion(self):
        pad_id, ls = SpecialTacticsTokens.PAD_ID, self.config.label_smoothing
        if self.config.loss == "par2":
            return Par2Loss(self.timeout, pad_id, ls)
        if self.config.loss == "ce":
            return CELoss(pad_id, ls)
        raise ValueError(f"Unknown loss: {self.config.loss}")


    def __scheduler(self, optimizer):
        total_steps = self.config.epochs * len(self.train)
        warmup_steps = self.config.warmup_steps
        if warmup_steps >= total_steps:
            raise ValueError(f"warmup_steps ({warmup_steps}) must be less than total steps ({total_steps})")

        if self.config.scheduler == "onecycle":
            kwargs = {"pct_start": warmup_steps / total_steps} if warmup_steps > 0 else {}
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.lr,
                total_steps=total_steps,
                **kwargs,
            )

        if self.config.scheduler != "cosine":
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
        )
        if warmup_steps == 0:
            return main_scheduler

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            total_iters=warmup_steps,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, main_scheduler],
            milestones=[warmup_steps],
        )


    def __train(self):
        device = self.model.device
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.decay)
        criterion = self.__criterion()
        scheduler = self.__scheduler(opt)
        min_loss = inf

        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0.0

            for smt, strat, bench in tqdm(self.train, desc=f"Epoch {epoch}"):
                opt.zero_grad()
                smt, strat = smt.to(device), strat.to(device)
                logits = self.model(smt, strat[:, :-1])
                loss = criterion(logits, strat[:, 1:], bench)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                train_loss += loss.item()

            val_loss = mean_loss(self.model, self.validation)
            log.info(f"Epoch {epoch}: train loss - {train_loss / len(self.train):.4f}, validation loss - {val_loss:.4f}, lr - {scheduler.get_last_lr()[0]:.6f}")
            if val_loss < min_loss:
                torch.save(self.__state_dict(), self.model_storage / f'{self.config.name}.pt')
                min_loss = val_loss
        torch.save(self.__state_dict(), self.model_storage / f'{self.config.name}-final.pt')

    def __state_dict(self):
        # torch.compile wraps the model and prefixes every key with "_orig_mod."
        return getattr(self.model, "_orig_mod", self.model).state_dict()

    def execute(self):
        self.__train()

