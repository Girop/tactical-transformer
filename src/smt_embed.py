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

GIN_BATCH_SIZE = 128


def _build_graph(args: tuple[Path, int]) -> dict | None:
    """Worker function for parallel graph construction."""
    path, timeout = args
    try:
        return build_smt_graph_dict_timeout(path, timeout)
    except:
        return None


def _graph_to_data(graph_dict, vocabulary):
    if graph_dict is None:
        return None
    data = graph_dict_to_gin_data(graph_dict, vocabulary)
    if data is None or data.num_nodes == 0:
        _suppress_z3_destructor_noise()
        return None
    return data


def _batched_gin_forward(data_items: list, selector: GraphSelector) -> list[torch.Tensor]:
    """Run GIN inference in batches."""
    results = []
    for i in range(0, len(data_items), GIN_BATCH_SIZE):
        batch = Batch.from_data_list(data_items[i:i + GIN_BATCH_SIZE]).to(selector.device)
        with torch.no_grad():
            embs = selector.model.forward_embedding(batch)
        for emb in embs:
            results.append(torch.from_numpy(normalize_gin_l2(emb.cpu().numpy())))
    return results


def embed_benchmarks(paths, cache_dir: Path = Path("data/embedding_cache")) -> dict[Path, torch.Tensor]:
    """GIN embedding per path; files that cannot be encoded are left out.

    Results are cached on disk, failed files as None so they are not retried.
    """
    cache = Path(cache_dir) / f"{GIN_MODEL_DIR.parent.name}-{GIN_MODEL_DIR.name}.pt"
    # Keyed by str: torch.load(weights_only=True) cannot unpickle Path objects.
    embeddings = {Path(p): e for p, e in torch.load(cache).items()} if cache.exists() else {}

    paths = set(map(Path, paths))
    todo = sorted(paths - embeddings.keys())
    if todo:
        selector = GraphSelector.load(GIN_MODEL_DIR)
        n_workers = min(len(todo), os.cpu_count() or 1)

        # Parallel graph construction
        work = [(p, selector.graph_timeout) for p in todo]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            graph_dicts = list(pool.map(_build_graph, work))

        # Convert to PyG data objects
        valid_paths = []
        valid_data = []
        for p, graph_dict in zip(todo, graph_dicts):
            data = _graph_to_data(graph_dict, selector.vocabulary)
            if data is not None:
                valid_paths.append(p)
                valid_data.append(data)
            else:
                embeddings[p] = None

        # Batched GIN forward pass
        if valid_data:
            embs = _batched_gin_forward(valid_data, selector)
            for p, emb in zip(valid_paths, embs):
                embeddings[p] = emb

        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({str(p): e for p, e in embeddings.items()}, cache)

    return {p: embeddings[p] for p in paths if embeddings[p] is not None}
