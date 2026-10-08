"""SMT-Select embeddings of SMT-LIB files: a frozen GIN over the term graph."""
import json
from pathlib import Path

import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from smt_select.models.graph.common import graph_dict_to_gin_data
from smt_select.models.graph.selector import GraphSelector
from smt_select.models.graph_text.selector import normalize_gin_l2
from smt_select.representations.graph_rep import _suppress_z3_destructor_noise, build_smt_graph_dict_timeout

# No QF_NIA model ships with smt-select; QF_NRA has (nearly) the same operator vocabulary.
GIN_MODEL_DIR = Path(__file__).resolve().parents[1] / "smt-select/models/gin_pwc/QF_NRA/seed0"
GIN_DIM = json.loads((GIN_MODEL_DIR / "config.json").read_text())["hidden_dim"]   # size of every embedding


def gin_embedding(path: Path, selector: GraphSelector) -> torch.Tensor | None:
    """L2-normalised graph embedding, or None if the file cannot be encoded."""
    graph_dict = build_smt_graph_dict_timeout(path, selector.graph_timeout)
    data = graph_dict_to_gin_data(graph_dict, selector.vocabulary) if graph_dict is not None else None
    del graph_dict
    if data is None or data.num_nodes == 0:
        _suppress_z3_destructor_noise()
        return None
    with torch.no_grad():
        emb = selector.model.forward_embedding(Batch.from_data_list([data]).to(selector.device))
    return torch.from_numpy(normalize_gin_l2(emb[0].cpu().numpy()))


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
        selector = GraphSelector.load(GIN_MODEL_DIR)   # dropout 0, eval mode
        for p in tqdm(todo, desc="GIN embeddings"):
            embeddings[p] = gin_embedding(p, selector)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({str(p): e for p, e in embeddings.items()}, cache)

    return {p: embeddings[p] for p in paths if embeddings[p] is not None}
