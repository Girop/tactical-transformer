import hashlib
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from smt_select.models.graph.common import graph_dict_to_gin_data
from smt_select.models.graph.selector import GraphSelector
from smt_select.models.graph_text.selector import GIN_L2_EPS
from smt_select.representations.graph_rep import (
    _suppress_z3_destructor_noise, build_smt_graph_dict_timeout)
from torch_geometric.data import Batch

log = logging.getLogger(__name__)

GIN_DIM = 64
GIN_MODELS_ROOT = Path(__file__).resolve().parents[1] / "smt-select/models/gin_pwc"
GRAPH_TIMEOUT = 5
FALLBACK_LOGIC = "QF_NRA"


def _build_graph(args: tuple[Path, int]) -> dict | None:
    path, timeout = args
    try:
        return build_smt_graph_dict_timeout(path, timeout)
    except:
        return None


class GINEmbedder:

    def __init__(self, gin_model: str = "matched", cache_dir: Path = Path("data/embedding_cache"), batch_size: int = 128):
        self.gin_model = gin_model
        self.batch_size = batch_size
        self._cache_dir = cache_dir / gin_model
        self._embeddings: dict[Path, torch.Tensor | None] = {}
        self._selectors: dict[str, GraphSelector] = {}
        self._available_logics = {d.name for d in GIN_MODELS_ROOT.iterdir() if d.is_dir()}
        self._load_cache()

    def _resolve_logic(self, path: Path) -> str:
        if self.gin_model != "matched":
            return self.gin_model
        logic = path.parts[2]
        return logic if logic in self._available_logics else FALLBACK_LOGIC

    def _get_selector(self, logic: str) -> GraphSelector:
        if logic not in self._selectors:
            model_dir = GIN_MODELS_ROOT / logic / "seed0"
            self._selectors[logic] = GraphSelector.load(model_dir)
            log.info(f"Loaded GIN model for {logic} ({model_dir})")
        return self._selectors[logic]

    def _entry_path(self, path: Path) -> Path:
        key = hashlib.sha256(str(path).encode()).hexdigest()
        return self._cache_dir / f"{key}.pt"

    def _load_cache(self):
        if not self._cache_dir.exists():
            return
        for f in self._cache_dir.glob("*.pt"):
            entry = torch.load(f, weights_only=False)
            for path_str, emb in entry.items():
                self._embeddings[Path(path_str)] = emb

    def _save_entry(self, path: Path, emb: torch.Tensor | None):
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({str(path): emb}, self._entry_path(path))

    def _batched_forward(self, data_items: list, selector: GraphSelector) -> list[torch.Tensor]:
        batch = Batch.from_data_list(data_items).to(selector.device)
        with torch.no_grad():
            embs = selector.model.forward_embedding(batch).cpu().numpy()
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms < GIN_L2_EPS, 1.0, norms)
        normalized = (embs / norms).astype(np.float32)
        return [torch.from_numpy(row) for row in normalized]

    def _embed_group(self, paths: list[Path], graphs: dict[Path, dict | None], selector: GraphSelector):
        pending_paths: list[Path] = []
        pending_data = []

        for p in paths:
            data = graph_dict_to_gin_data(graphs[p], selector.vocabulary) if graphs[p] is not None else None
            if data is not None and data.num_nodes > 0:
                pending_paths.append(p)
                pending_data.append(data)
                if len(pending_data) >= self.batch_size:
                    for pp, emb in zip(pending_paths, self._batched_forward(pending_data, selector)):
                        self._embeddings[pp] = emb
                        self._save_entry(pp, emb)
                    pending_paths.clear()
                    pending_data.clear()
            else:
                if data is not None:
                    _suppress_z3_destructor_noise()
                self._embeddings[p] = None
                self._save_entry(p, None)

        if pending_data:
            for pp, emb in zip(pending_paths, self._batched_forward(pending_data, selector)):
                self._embeddings[pp] = emb
                self._save_entry(pp, emb)

    def embed(self, paths) -> dict[Path, torch.Tensor]:
        """GIN embedding per path; files that cannot be encoded are left out."""
        paths = set(map(Path, paths))
        todo = sorted(paths - self._embeddings.keys())

        if todo:
            graphs: dict[Path, dict | None] = {}
            with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
                futures = {pool.submit(_build_graph, (p, GRAPH_TIMEOUT)): p for p in todo}
                for future in as_completed(futures):
                    graphs[futures[future]] = future.result()

            by_logic: dict[str, list[Path]] = defaultdict(list)
            for p in todo:
                by_logic[self._resolve_logic(p)].append(p)

            for logic, logic_paths in by_logic.items():
                self._embed_group(logic_paths, graphs, self._get_selector(logic))

        return {p: self._embeddings[p] for p in paths if self._embeddings.get(p) is not None}
