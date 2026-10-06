import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import z3
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from z3 import z3consts

# Fixed, logic-agnostic id spaces: every z3 operator kind and sort kind, plus specials.
OP_KINDS = sorted({v for k, v in vars(z3consts).items() if k.startswith("Z3_OP_")})
SORT_KINDS = sorted({v for k, v in vars(z3consts).items() if k.endswith("_SORT") and k.startswith("Z3_")})
ROOT, QUANTIFIER, UNK = len(OP_KINDS), len(OP_KINDS) + 1, len(OP_KINDS) + 2
NUM_OPS = len(OP_KINDS) + 3          # size of the operator embedding table
NUM_SORTS = len(SORT_KINDS) + 1      # last id: unknown sort / root
_OP_INDEX = {k: i for i, k in enumerate(OP_KINDS)}
_SORT_INDEX = {k: i for i, k in enumerate(SORT_KINDS)}


@dataclass
class SmtGraph:
    op: np.ndarray         # [N] operator id (index into OP_KINDS, or ROOT/QUANTIFIER/UNK)
    sort: np.ndarray       # [N] sort id (index into SORT_KINDS)
    depth: np.ndarray      # [N] distance from the root
    parents: np.ndarray    # [N] how many terms use this node (sharing)
    value: np.ndarray      # [N] numerals: sign * log1p(|value|), else 0
    truncated: np.ndarray  # [N] 1 if some children were cut by max_nodes
    edges: np.ndarray      # [E, 2] (parent, child)
    arg: np.ndarray        # [E] argument position of the child
    total_nodes: int       # nodes before truncation

    def distances(self, max_dist: int = 8) -> np.ndarray:
        """[N, N] undirected hop distance, capped at ``max_dist`` (also for unreachable pairs)."""
        n = len(self.op)
        adj = csr_matrix((np.ones(len(self.edges)), (self.edges[:, 0], self.edges[:, 1])), shape=(n, n))
        d = dijkstra(adj, directed=False, unweighted=True, limit=max_dist)
        return np.minimum(np.nan_to_num(d, posinf=max_dist), max_dist).astype(np.uint8)


def _numeral(e) -> float:
    if z3.is_int_value(e) or z3.is_bv_value(e):
        v = e.as_long()
    elif z3.is_rational_value(e):
        v = e.numerator_as_long() / e.denominator_as_long()
    else:
        return 0.0
    return math.copysign(math.log1p(abs(v)), v)


def _children(e) -> list:
    if z3.is_app(e):
        return e.children()
    return [e.body()] if z3.is_quantifier(e) else []   # bound variables are leaves


def _count_terms(roots) -> int:
    seen, stack = set(), list(roots)
    while stack:
        e = stack.pop()
        if e.get_id() not in seen:
            seen.add(e.get_id())
            stack += _children(e)
    return len(seen)


def parse_graph(path: str | Path, max_nodes: int = 2048) -> SmtGraph:
    """Term DAG of an SMT-LIB file, breadth-first from a virtual root over its assertions."""
    ctx = z3.Context()   # a fresh context per file keeps z3's term table from growing
    assertions = list(z3.parse_smt2_file(str(path), ctx=ctx))

    index: dict[int, int] = {}   # z3 term id -> node index
    op, sort, depth, value, parents, truncated = [ROOT], [NUM_SORTS - 1], [0], [0.0], [0], [0]
    edges, arg = [], []
    frontier = [(0, i, a) for i, a in enumerate(assertions)]   # (parent node, arg position, term)
    while frontier:
        nxt = []
        for p, pos, e in frontier:
            node = index.get(e.get_id())
            if node is None:
                if len(op) >= max_nodes:
                    truncated[p] = 1
                    continue
                node = index[e.get_id()] = len(op)
                op.append(_OP_INDEX.get(e.decl().kind(), UNK) if z3.is_app(e) else QUANTIFIER if z3.is_quantifier(e) else UNK)
                sort.append(_SORT_INDEX.get(e.sort().kind(), NUM_SORTS - 1))
                depth.append(depth[p] + 1)
                value.append(_numeral(e))
                parents.append(0)
                truncated.append(0)
                nxt += [(node, i, c) for i, c in enumerate(_children(e))]
            edges.append((p, node))
            arg.append(pos)
            parents[node] += 1
        frontier = nxt

    return SmtGraph(
        op=np.array(op, dtype=np.int16), sort=np.array(sort, dtype=np.int8),
        depth=np.array(depth, dtype=np.int16), parents=np.array(parents, dtype=np.int32),
        value=np.array(value, dtype=np.float32), truncated=np.array(truncated, dtype=np.int8),
        edges=np.array(edges, dtype=np.int32).reshape(-1, 2), arg=np.array(arg, dtype=np.int16),
        total_nodes=_count_terms(assertions) + 1,
    )


def load_graph(path: str | Path, cache_dir: str | Path = "data/graph_cache", max_nodes: int = 2048) -> SmtGraph:
    """``parse_graph`` with an on-disk cache keyed by file path and ``max_nodes``."""
    key = hashlib.sha1(f"{Path(path).as_posix()}:{max_nodes}".encode()).hexdigest()
    cached = Path(cache_dir) / f"{key}.npz"
    if cached.exists():
        with np.load(cached) as d:
            return SmtGraph(**{k: d[k] for k in d.files if k != "total_nodes"}, total_nodes=int(d["total_nodes"]))
    g = parse_graph(path, max_nodes)
    cached.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cached, **{k: v for k, v in vars(g).items()})
    return g


_OP_NAMES = {v: k[6:].lower() for k, v in vars(z3consts).items() if k.startswith("Z3_OP_")}
_OP_NAMES.update({z3consts.Z3_OP_UNINTERPRETED: "var", z3consts.Z3_OP_ANUM: "num"})


def node_label(g: SmtGraph, i: int) -> str:
    """e.g. ``mul``, ``var``, ``num +1.61`` (numerals show sign * log1p(|value|))."""
    op = int(g.op[i])
    name = {ROOT: "ROOT", QUANTIFIER: "quantifier", UNK: "?"}.get(op) or _OP_NAMES.get(OP_KINDS[op], "?")
    if name == "num":
        name += f" {g.value[i]:+.2f}"
    return name + (" ..." if g.truncated[i] else "")


def to_text(g: SmtGraph) -> str:
    """Indented tree from the root; a shared node is expanded once, later shown as ``-> #id``."""
    children: dict[int, list[int]] = {}
    for parent, child in g.edges.tolist():
        children.setdefault(parent, []).append(child)
    lines, seen, stack = [], set(), [(0, 0)]
    while stack:
        node, level = stack.pop()
        if node in seen:
            lines.append(f"{'   ' * level}-> #{node} {node_label(g, node)}")
            continue
        seen.add(node)
        lines.append(f"{'   ' * level}#{node} {node_label(g, node)}")
        stack += [(c, level + 1) for c in reversed(children.get(node, []))]
    return "\n".join(lines)


def to_dot(g: SmtGraph) -> str:
    """Graphviz DOT (render with e.g. ``dot -Tsvg``); edges are labelled with the argument position."""
    lines = ["digraph smt {", "  node [shape=box, fontname=monospace];"]
    lines += [f'  n{i} [label="#{i} {node_label(g, i)}"];' for i in range(len(g.op))]
    lines += [f'  n{p} -> n{c} [label="{a}"];' for (p, c), a in zip(g.edges.tolist(), g.arg.tolist())]
    return "\n".join(lines + ["}"])


if __name__ == "__main__":   # python smt_graph.py FILE.smt2 [--dot] [--max-nodes N]
    import argparse

    parser = argparse.ArgumentParser(description="Print the term graph of an SMT-LIB file")
    parser.add_argument("file")
    parser.add_argument("--dot", action="store_true", help="Graphviz DOT instead of an indented tree")
    parser.add_argument("--max-nodes", type=int, default=2048)
    args = parser.parse_args()
    graph = parse_graph(args.file, args.max_nodes)
    print(to_dot(graph) if args.dot else to_text(graph))
