"""SMT-Select embeddings of SMT-LIB files: a frozen GIN over the term graph."""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from torch_geometric.data import Batch

from smt_select.models.graph.common import graph_dict_to_gin_data
from smt_select.models.graph.selector import GraphSelector
from smt_select.models.graph_text.selector import normalize_gin_l2
from smt_select.representations.graph_rep import _suppress_z3_destructor_noise, build_smt_graph_dict_timeout

# TODO: No QF_NIA model ships with smt-select; QF_NRA has (nearly) the same operator vocabulary.
GIN_MODEL_DIR = Path(__file__).resolve().parents[1] / "smt-select/models/gin_pwc/QF_NRA/seed0"
GIN_DIM = json.loads((GIN_MODEL_DIR / "config.json").read_text())["hidden_dim"]   # size of every embedding


def _build_graph(args: tuple[Path, int]) -> dict | None:
    path, timeout = args
    try:
        return build_smt_graph_dict_timeout(path, timeout)
    except:
        return None


class GINEmbedder:

    def __init__(self, model_dir: Path = GIN_MODEL_DIR, cache_dir: Path = Path("data/embedding_cache"), batch_size: int = 128):
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._cache_path = cache_dir / f"{model_dir.parent.name}-{model_dir.name}.pt"
        self._embeddings: dict[Path, torch.Tensor | None] = {}
        self._load_cache()

    def _load_cache(self):
        if self._cache_path.exists():
            self._embeddings = {Path(p): e for p, e in torch.load(self._cache_path).items()}

    def _save_cache(self):
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({str(p): e for p, e in self._embeddings.items()}, self._cache_path)

    def _graph_to_data(self, graph_dict, vocabulary):
        if graph_dict is None:
            return None
        data = graph_dict_to_gin_data(graph_dict, vocabulary)
        if data is None or data.num_nodes == 0:
            _suppress_z3_destructor_noise()
            return None
        return data

    def _batched_forward(self, data_items: list, selector: GraphSelector) -> list[torch.Tensor]:
        results = []
        for i in range(0, len(data_items), self.batch_size):
            batch = Batch.from_data_list(data_items[i:i + self.batch_size]).to(selector.device)
            with torch.no_grad():
                embs = selector.model.forward_embedding(batch)
            for emb in embs:
                results.append(torch.from_numpy(normalize_gin_l2(emb.cpu().numpy())))
        return results

    def embed(self, paths) -> dict[Path, torch.Tensor]:
        """GIN embedding per path; files that cannot be encoded are left out."""
        paths = set(map(Path, paths))
        todo = sorted(paths - self._embeddings.keys())
        if todo:
            selector = GraphSelector.load(self.model_dir)
            n_workers = min(len(todo), os.cpu_count() or 1)

            work = [(p, selector.graph_timeout) for p in todo]
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                graph_dicts = list(pool.map(_build_graph, work))

            valid_paths = []
            valid_data = []
            for p, graph_dict in zip(todo, graph_dicts):
                data = self._graph_to_data(graph_dict, selector.vocabulary)
                if data is not None:
                    valid_paths.append(p)
                    valid_data.append(data)
                else:
                    self._embeddings[p] = None

            if valid_data:
                embs = self._batched_forward(valid_data, selector)
                for p, emb in zip(valid_paths, embs):
                    self._embeddings[p] = emb

            self._save_cache()

        return {p: self._embeddings[p] for p in paths if self._embeddings.get(p) is not None}
