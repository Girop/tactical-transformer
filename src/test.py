from pydantic.dataclasses import dataclass
from dataloader import Status, Benchmark
from collections import defaultdict
from dataclasses import asdict
import logging
from tqdm import tqdm
from pathlib import Path
from utils import ExperimentConfig
from transformer import ModelConfig
from torch import nn
from tactics import CATALOG
from z3alpha.evaluator import SolverRunner
import json
from torch.utils.data import DataLoader


log = logging.getLogger(__name__)


def par2(time_s: float, solved: bool, timeout: float) -> float:
    return time_s if solved else timeout * 2


@dataclass(frozen=True)
class LogicStats:
    logic: str
    model_sr: float
    baseline_sr: float
    model_par2: float
    baseline_par2: float
    n: int

    @classmethod
    def from_benchmarks(cls, model_benchmarks: list[Benchmark], base_benchmarks: list[Benchmark], logic: str, timeout: float) -> LogicStats:
        def sr_par2(benchmarks: list[Benchmark]):
            n = max(len(benchmarks), 1)
            return sum(b.solved for b in benchmarks) / n, sum(par2(b.time_s, b.solved, timeout) for b in benchmarks) / n
        m_sr, m_p2 = sr_par2(model_benchmarks)
        b_sr, b_p2 = sr_par2(base_benchmarks)
        return cls(logic=logic, model_sr=m_sr, baseline_sr=b_sr, model_par2=m_p2, baseline_par2=b_p2, n=len(model_benchmarks))


@dataclass(frozen=True)
class TestResults:
    overall: LogicStats
    per_logic: list[LogicStats]

    @classmethod
    def from_benchmarks(cls, model_results: list[Benchmark], baseline: list[Benchmark], timeout: float = 10.0) -> TestResults:
        overall = LogicStats.from_benchmarks(model_results, baseline, "Overall", timeout)

        model_by_logic: dict[str, list[Benchmark]] = defaultdict(list)
        baseline_by_logic: dict[str, list[Benchmark]] = defaultdict(list)
        for b in model_results:
            model_by_logic[b.logic].append(b)
        for b in baseline:
            baseline_by_logic[b.logic].append(b)

        per_logic = [
            LogicStats.from_benchmarks(model_by_logic.get(logic, []), baseline_by_logic.get(logic, []), logic, timeout)
            for logic in sorted(set(model_by_logic) | set(baseline_by_logic))
        ]
        return cls(overall=overall, per_logic=per_logic)

    def display(self):
        rows = [self.overall, *self.per_logic]
        w = max((len(r.logic) for r in rows), default=10) + 2
        header = f"{'─' * (w + 42)}"
        log.info(f"\n{header}")
        log.info(f"  {'':>{w}} {'Model SR':>9} {'Model P2':>9} {'Base SR':>9} {'Base P2':>9}  {'n':>4}")
        log.info(header)
        for r in rows:
            log.info(f"  {r.logic:>{w}} {r.model_sr:>9.2f} {r.model_par2:>9.2f} {r.baseline_sr:>9.2f} {r.baseline_par2:>9.2f}  {r.n:>4}")
        log.info(header)


    def to_json(self, path: Path, model: ModelConfig, experiment: ExperimentConfig):
        data = {
            "model": str(model),
            "experiment": str(experiment),
            "overall": asdict(self.overall),
            "per_logic": [asdict(r) for r in self.per_logic],
        }
        path.write_text(json.dumps(data, indent=2))


def generate_strategies(model: nn.Module, data: DataLoader) -> list[tuple[Path, str]]:
    device = model.device
    strats = []
    for smt, _, bench in tqdm(data, desc="Generating"):
        ids = model.generate(smt.to(device)).tolist()
        strats.extend((b.benchmark, CATALOG.tactics_to_text(id_)) for b, id_ in zip(bench, ids))
    return strats


# Cache by: (benchmark, strategy, timeout), values: (status, runtime)
def run_solver_on_benchmarks(benchmarks: list[tuple[Path, str]], z3path: Path | str, timeout=10.0, cache: Path = Path("data/benchmark_cache")) -> list[Benchmark]:
    cache.mkdir(exist_ok=True)
    cache_file = cache / "results.jsonl"

    store: dict = {}
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            if line:
                entry = json.loads(line)
                store[entry["key"]] = {"status": entry["status"], "runtime": entry["runtime"]}

    results = []
    with cache_file.open("a") as f:
        for idx, (bench_path, strategy) in enumerate(tqdm(benchmarks, desc="Testing")):
            key = f"{bench_path}|{strategy}|{timeout}"
            if key in store:
                status, runtime = store[key]["status"], store[key]["runtime"]
            else:
                _, status, runtime, _ = SolverRunner(
                    z3path,
                    str(bench_path),
                    timeout,
                    run_id=idx,
                    z3_strategy=strategy
                ).execute()
                store[key] = {"status": status, "runtime": runtime}
                f.write(json.dumps({"key": key, "status": status, "runtime": runtime}) + "\n")
                f.flush()
            results.append(Benchmark(strategy, bench_path, Status[status], float(runtime), status in ("sat", "unsat")))
    return results


def run_test(model: nn.Module, test: DataLoader, z3path: Path | str, timeout: float = 10.0) -> TestResults:
    baseline = test.dataset.bench_data
    assert len(baseline) == len(set(b.benchmark for b in baseline))
    strat_bench = generate_strategies(model, test)
    model_results = run_solver_on_benchmarks(strat_bench, z3path, timeout)
    return TestResults.from_benchmarks(model_results, baseline, timeout)


def test(model: nn.Module, test: DataLoader, z3: Path, model_config, experiment):
    results = run_test(model, test, z3)
    results.display()
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)
    results.to_json(out_dir / f"{experiment.name}-out.json", model_config, experiment)

