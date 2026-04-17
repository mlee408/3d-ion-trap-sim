from __future__ import annotations

"""
automate.py

First-pass automation layer for geometry search / batch evaluation.

Design goals
------------
- Keep `run_case.py` as the single-case evaluator.
- Let this file orchestrate many runs, score them, and rank results.
- Stay lightweight: use subprocess calls first, so it works even if run_case.py
  is still mostly CLI-oriented.
- Be easy to extend toward smarter optimization later.

Typical usage
-------------
1. Provide a mesh generator command template that can turn parameters into a mesh.
2. Provide RF / ground tags and physical parameters for run_case.py.
3. Run a random search over parameter bounds.
4. Collect JSON reports and a CSV summary.

Example (run from src/ directory)
-------
python automate.py \
  --run-case ./run_case.py \
  --mesh-template "python ../meshes/run_case.py \
    --rf ../meshes/step/rf.step \
    --dc ../meshes/step/dc.step \
    --ground ../meshes/step/ground.step \
    --lc-electrode {lc_electrode} \
    --lc-center {lc_center} \
    --lc-far {lc_far} \
    --pad-z-top {pad_z_top} \
    --nopopup \
    --out {mesh_path}" \
  --workdir ./sweep_000 \
  --rf-tags 1 \
  --ground-tags 3 \
  --outer-tags 4 \
  --param lc_electrode:0.002:0.008 \
  --param lc_center:0.003:0.010 \
  --param lc_far:0.020:0.060 \
  --param pad_z_top:0.300:0.800 \
  --degree 2 \
  --mass-amu 40.0 \
  --charge-e 1.0 \
  --rf-freq 40e6 \
  --vrf 150 \
  --coord-unit 1e-3 \
  --n-cases 20 \
  --seed 42
"""

import argparse
import csv
import itertools
import json
import math
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------


@dataclass
class ParamSpec:
    name: str
    low: float
    high: float

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)


@dataclass
class RunConfig:
    run_case_py: Path
    workdir: Path
    mesh_template: str
    rf_tags: List[int]
    ground_tags: List[int]
    basis_tags: List[int]
    degree: int
    rf_freq: float
    mass_amu: float
    charge_e: float
    h: float
    depth_ray_length: float
    depth_nrays: int
    prefix: str
    vrf: float
    coord_unit: Optional[float]
    no_depth: bool
    # RF-null search bounds (mesh units); None means no bound / use run_case.py default
    r0_z_min: Optional[float]
    r0_z_max: Optional[float]
    r0_x_min: Optional[float]
    r0_x_max: Optional[float]
    r0_y_min: Optional[float]
    r0_y_max: Optional[float]
    r0_search_margin: Optional[float]
    r0_x_auto: bool
    # Outer (far-field / Neumann) boundary tags — do NOT include in ground_tags
    outer_tags: List[int]


@dataclass
class CaseResult:
    case_id: str
    params: Dict[str, float]
    mesh_path: str
    case_dir: str
    status: str
    score: Optional[float]
    depth_eV: Optional[float]
    min_freq_hz: Optional[float]
    max_freq_hz: Optional[float]
    mode_spread_hz: Optional[float]
    center_offset_m: Optional[float]
    report_path: Optional[str]
    stderr_path: Optional[str]
    stdout_path: Optional[str]
    error_message: Optional[str]
    elapsed_s: float


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def parse_param_specs(raw_specs: Sequence[str]) -> List[ParamSpec]:
    """Parse repeated CLI args like: rf_height:40:120"""
    specs: List[ParamSpec] = []
    for raw in raw_specs:
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --param '{raw}'. Expected format name:low:high"
            )
        name, low_s, high_s = parts
        low = float(low_s)
        high = float(high_s)
        if high < low:
            raise ValueError(f"Invalid bounds for {name}: high < low")
        specs.append(ParamSpec(name=name, low=low, high=high))
    return specs


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())

def find_paths(obj, prefix="root"):
    if isinstance(obj, Path):
        print(prefix, "->", obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            find_paths(v, f"{prefix}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_paths(v, f"{prefix}[{i}]")



def render_template(template: str, mapping: Dict[str, Any]) -> str:
    try:
        return template.format(**mapping)
    except KeyError as e:
        missing = str(e)
        raise KeyError(
            f"Template references missing field {missing}. Available keys: {sorted(mapping)}"
        ) from e


def append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    mkdir(path.parent)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# -----------------------------------------------------------------------------
# Candidate generation
# -----------------------------------------------------------------------------


def sample_random_params(specs: Sequence[ParamSpec], rng: random.Random) -> Dict[str, float]:
    return {spec.name: spec.sample(rng) for spec in specs}


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------


def extract_metrics_from_report(report: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract the subset of metrics that the scorer needs.

    This is written defensively because report schemas often change slightly.
    """
    depth_eV = None
    min_freq_hz = None
    max_freq_hz = None
    mode_spread_hz = None
    center_offset_m = None

    # Depth
    depth = report.get("depth")
    if isinstance(depth, dict):
        depth_eV = safe_float(depth.get("depth_eV"))

    # Frequencies
    secular = report.get("secular") or report.get("secular_frequencies")
    if isinstance(secular, dict):
        freq_hz = secular.get("freq_hz") or secular.get("frequencies_hz")
        if isinstance(freq_hz, list) and freq_hz:
            freq_vals = [safe_float(x) for x in freq_hz]
            freq_vals = [x for x in freq_vals if x is not None]
            if freq_vals:
                min_freq_hz = min(freq_vals)
                max_freq_hz = max(freq_vals)
                mode_spread_hz = max_freq_hz - min_freq_hz

    # Trap center offset from bbox center or origin if not available
    trap_min = report.get("trap_min") or {}
    r0_si = report.get("r0_SI_m")
    if isinstance(r0_si, list) and r0_si:
        # Default: norm from origin. Later you can replace this with distance
        # from expected design axis or from linear-trap center.
        vals = [safe_float(x) for x in r0_si]
        vals = [x for x in vals if x is not None]
        if vals:
            center_offset_m = math.sqrt(sum(x * x for x in vals))

    return {
        "depth_eV": depth_eV,
        "min_freq_hz": min_freq_hz,
        "max_freq_hz": max_freq_hz,
        "mode_spread_hz": mode_spread_hz,
        "center_offset_m": center_offset_m,
    }



def score_case_metrics(metrics: Dict[str, Optional[float]]) -> Optional[float]:
    """Simple first-pass scalar objective.

    Current intent:
    - reward deeper traps,
    - reward stronger weakest-mode confinement,
    - penalize mode asymmetry,
    - penalize large center offsets.

    Tune these weights after you inspect a few dozen runs.
    """
    depth_eV = metrics.get("depth_eV")
    min_freq_hz = metrics.get("min_freq_hz")
    mode_spread_hz = metrics.get("mode_spread_hz")
    center_offset_m = metrics.get("center_offset_m")

    if depth_eV is None or min_freq_hz is None:
        return None

    score = 0.0
    score += 5.0 * depth_eV
    score += 1e-6 * min_freq_hz

    if mode_spread_hz is not None:
        score -= 2e-6 * mode_spread_hz
    if center_offset_m is not None:
        score -= 1e6 * center_offset_m

    return score


# -----------------------------------------------------------------------------
# External execution
# -----------------------------------------------------------------------------


def run_subprocess(command: Sequence[str], *, cwd: Optional[Path], stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("w") as fout, stderr_path.open("w") as ferr:
        proc = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            stdout=fout,
            stderr=ferr,
            text=True,
        )
    return int(proc.returncode)



def generate_mesh(
    params: Dict[str, float],
    *,
    case_dir: Path,
    mesh_template: str,
) -> Path:
    """Run the user-provided mesh-generation command template.

    The template may reference:
    - all parameter names, e.g. {rf_height}
    - {case_dir}
    - {mesh_path}
    """
    mesh_path = (case_dir / "mesh.msh").resolve()
    mapping: Dict[str, Any] = dict(params)
    mapping["case_dir"] = str(case_dir.resolve())
    mapping["mesh_path"] = str(mesh_path)

    command_str = render_template(mesh_template, mapping)
    cmd = shlex.split(command_str)

    gen_stdout = case_dir / "meshgen.stdout.txt"
    gen_stderr = case_dir / "meshgen.stderr.txt"
    rc = run_subprocess(cmd, cwd=None, stdout_path=gen_stdout, stderr_path=gen_stderr)
    if rc != 0:
        raise RuntimeError(
            f"Mesh generation failed with exit code {rc}. See {gen_stderr}"
        )

    if not mesh_path.exists():
        # Allow generators that output xdmf directly if template uses {mesh_path}
        # with another suffix. In that case user should adapt this function.
        xdmf_candidate = case_dir / "mesh.xdmf"
        if xdmf_candidate.exists():
            return xdmf_candidate
        raise FileNotFoundError(
            f"Expected mesh at {mesh_path} (or mesh.xdmf), but neither exists."
        )

    return mesh_path



def build_run_case_command(cfg: RunConfig, mesh_path: Path, case_dir: Path, case_prefix: str) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(cfg.run_case_py.resolve()),
        "--mesh", str(mesh_path.resolve()),
        "--outdir", str(case_dir.resolve()),
        "--degree", str(cfg.degree),
        "--rf-freq", str(cfg.rf_freq),
        "--mass-amu", str(cfg.mass_amu),
        "--charge-e", str(cfg.charge_e),
        "--h", str(cfg.h),
        "--depth-ray-length", str(cfg.depth_ray_length),
        "--depth-nrays", str(cfg.depth_nrays),
        "--prefix", case_prefix,
        "--vrf", str(cfg.vrf),
    ]

    if cfg.coord_unit is not None:
        cmd.extend(["--coord-unit", str(cfg.coord_unit)])
    if cfg.no_depth:
        cmd.append("--no-depth")

    # RF-null search bounds
    if cfg.r0_z_min is not None:
        cmd.extend(["--r0-z-min", str(cfg.r0_z_min)])
    if cfg.r0_z_max is not None:
        cmd.extend(["--r0-z-max", str(cfg.r0_z_max)])
    if cfg.r0_x_min is not None:
        cmd.extend(["--r0-x-min", str(cfg.r0_x_min)])
    if cfg.r0_x_max is not None:
        cmd.extend(["--r0-x-max", str(cfg.r0_x_max)])
    if cfg.r0_y_min is not None:
        cmd.extend(["--r0-y-min", str(cfg.r0_y_min)])
    if cfg.r0_y_max is not None:
        cmd.extend(["--r0-y-max", str(cfg.r0_y_max)])
    if cfg.r0_search_margin is not None:
        cmd.extend(["--r0-search-margin", str(cfg.r0_search_margin)])
    if cfg.r0_x_auto:
        cmd.append("--r0-x-auto")

    cmd.append("--rf-tags")
    cmd.extend(str(t) for t in cfg.rf_tags)

    cmd.append("--ground-tags")
    cmd.extend(str(t) for t in cfg.ground_tags)

    if cfg.basis_tags:
        cmd.append("--basis-tags")
        cmd.extend(str(t) for t in cfg.basis_tags)

    if cfg.outer_tags:
        cmd.append("--outer-tags")
        cmd.extend(str(t) for t in cfg.outer_tags)

    return cmd



def infer_report_path(case_dir: Path, case_prefix: str) -> Optional[Path]:
    candidates = [
        case_dir / f"{case_prefix}_report.json",
        case_dir / "report.json",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Fallback: search for a single json file in case_dir, excluding params.json
    # (which is always written by automate.py before run_case.py runs, so it
    # would otherwise be picked up on failures as a false-positive "report").
    jsons = [p for p in sorted(case_dir.glob("*.json")) if p.name != "params.json"]
    if len(jsons) == 1:
        return jsons[0]
    return None


# -----------------------------------------------------------------------------
# Single-case evaluation
# -----------------------------------------------------------------------------


def evaluate_case(
    case_index: int,
    params: Dict[str, float],
    *,
    cfg: RunConfig,
) -> CaseResult:
    case_id = f"case_{case_index:04d}"
    case_dir = mkdir(cfg.workdir / case_id)
    case_prefix = f"{cfg.prefix}_{case_id}"

    metadata_path = case_dir / "params.json"
    write_json(metadata_path, {"case_id": case_id, "params": params})

    start = time.time()
    stdout_path = case_dir / "run_case.stdout.txt"
    stderr_path = case_dir / "run_case.stderr.txt"

    try:
        mesh_path = generate_mesh(params, case_dir=case_dir, mesh_template=cfg.mesh_template)

        cmd = build_run_case_command(cfg, mesh_path=mesh_path, case_dir=case_dir, case_prefix=case_prefix)
        rc = run_subprocess(cmd, cwd=case_dir, stdout_path=stdout_path, stderr_path=stderr_path)
        if rc != 0:
            raise RuntimeError(f"run_case.py failed with exit code {rc}")

        report_path = infer_report_path(case_dir, case_prefix)
        if report_path is None:
            raise FileNotFoundError("Could not locate JSON report from run_case.py")

        report = load_json(report_path)
        metrics = extract_metrics_from_report(report)
        score = score_case_metrics(metrics)

        elapsed = time.time() - start
        return CaseResult(
            case_id=case_id,
            params=params,
            mesh_path=str(mesh_path),
            case_dir=str(case_dir),
            status="ok",
            score=score,
            depth_eV=metrics.get("depth_eV"),
            min_freq_hz=metrics.get("min_freq_hz"),
            max_freq_hz=metrics.get("max_freq_hz"),
            mode_spread_hz=metrics.get("mode_spread_hz"),
            center_offset_m=metrics.get("center_offset_m"),
            report_path=str(report_path),
            stderr_path=str(stderr_path),
            stdout_path=str(stdout_path),
            error_message=None,
            elapsed_s=elapsed,
        )
    except Exception as e:
        elapsed = time.time() - start
        return CaseResult(
            case_id=case_id,
            params=params,
            mesh_path=str(case_dir / "mesh.msh"),
            case_dir=str(case_dir),
            status="failed",
            score=None,
            depth_eV=None,
            min_freq_hz=None,
            max_freq_hz=None,
            mode_spread_hz=None,
            center_offset_m=None,
            report_path=None,
            stderr_path=str(stderr_path),
            stdout_path=str(stdout_path),
            error_message=str(e),
            elapsed_s=elapsed,
        )


# -----------------------------------------------------------------------------
# Search loop
# -----------------------------------------------------------------------------


def _result_row(result: CaseResult) -> Dict[str, Any]:
    return {
        "case_id": result.case_id,
        "status": result.status,
        "score": result.score,
        "depth_eV": result.depth_eV,
        "min_freq_hz": result.min_freq_hz,
        "max_freq_hz": result.max_freq_hz,
        "mode_spread_hz": result.mode_spread_hz,
        "center_offset_m": result.center_offset_m,
        "mesh_path": result.mesh_path,
        "case_dir": result.case_dir,
        "report_path": result.report_path,
        "stderr_path": result.stderr_path,
        "stdout_path": result.stdout_path,
        "error_message": result.error_message,
        "elapsed_s": result.elapsed_s,
        **{f"param_{k}": v for k, v in result.params.items()},
    }


def _load_existing_results(summary_csv: Path) -> Dict[str, Dict[str, Any]]:
    if not summary_csv.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with summary_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("case_id")
            if cid:
                out[cid] = row
    return out


def run_parallel_random_search(
    cfg: RunConfig,
    param_specs: Sequence[ParamSpec],
    *,
    n_cases: int,
    seed: int,
    max_workers: int,
    resume: bool,
) -> List[CaseResult]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    rng = random.Random(seed)
    results: List[CaseResult] = []

    summary_csv = cfg.workdir / "summary.csv"
    all_jsonl = cfg.workdir / "summary.jsonl"

    existing = _load_existing_results(summary_csv) if resume else {}

    # Pre-sample parameters deterministically from the provided seed so resumed
    # runs reproduce the same candidate set/order.
    planned: List[Tuple[int, Dict[str, float]]] = []
    for i in range(n_cases):
        params = sample_random_params(param_specs, rng)
        planned.append((i, params))

    pending: List[Tuple[int, Dict[str, float]]] = []
    for i, params in planned:
        case_id = f"case_{i:04d}"
        if resume and case_id in existing:
            continue
        pending.append((i, params))

    if resume and existing:
        print(f"Resume enabled: found {len(existing)} existing case rows in {summary_csv}")
        print(f"Skipping {len(planned) - len(pending)} already-recorded cases")

    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    def _submit(executor: ProcessPoolExecutor, item: Tuple[int, Dict[str, float]]):
        idx, params = item
        return executor.submit(evaluate_case, idx, params, cfg=cfg)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        future_map = {}
        pending_iter = iter(pending)

        # Fill the worker pool initially
        for _ in range(min(max_workers, len(pending))):
            try:
                item = next(pending_iter)
            except StopIteration:
                break
            fut = _submit(ex, item)
            future_map[fut] = item[0]

        while future_map:
            for fut in as_completed(list(future_map.keys()), timeout=None):
                _ = future_map.pop(fut)
                result = fut.result()
                results.append(result)

                row = _result_row(result)
                append_csv_row(summary_csv, row)
                with all_jsonl.open("a") as f:
                    f.write(json.dumps(row) + "")

                status = result.status
                score = result.score
                print(
                    f"[{result.case_id}] {status} | score={score} | elapsed={result.elapsed_s:.1f}s | {result.case_dir}"
                )

                try:
                    item = next(pending_iter)
                except StopIteration:
                    item = None
                if item is not None:
                    new_fut = _submit(ex, item)
                    future_map[new_fut] = item[0]
                break

    return results


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------



def print_best(results: Sequence[CaseResult], *, top_k: int = 5) -> None:
    valid = [r for r in results if r.status == "ok" and r.score is not None]
    valid.sort(key=lambda r: r.score if r.score is not None else -1e99, reverse=True)

    print("\nTop results")
    print("===========")
    if not valid:
        print("No successful cases with valid scores.")
        return

    for r in valid[:top_k]:
        print(
            f"{r.case_id}: score={r.score:.6g}, depth_eV={r.depth_eV}, "
            f"min_freq_hz={r.min_freq_hz}, mode_spread_hz={r.mode_spread_hz}, "
            f"case_dir={r.case_dir}"
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------



def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Batch driver / first-pass optimizer around run_case.py")

    ap.add_argument("--run-case", type=Path, required=True, help="Path to run_case.py")
    ap.add_argument("--mesh-template", type=str, required=True,
                    help="Command template for mesh generation. May reference params plus {case_dir} and {mesh_path}")
    ap.add_argument("--workdir", type=Path, required=True)

    ap.add_argument("--rf-tags", type=int, nargs="+", required=True)
    ap.add_argument("--ground-tags", type=int, nargs="+", required=True)
    ap.add_argument("--basis-tags", type=int, nargs="*", default=[])

    ap.add_argument("--param", action="append", default=[],
                    help="Parameter range in format name:low:high. Repeat for each parameter.")
    ap.add_argument("--n-cases", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-workers", type=int, default=8,
                    help="Number of concurrent case evaluations to run in separate processes.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip cases already recorded in summary.csv for this workdir.")

    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--rf-freq", type=float, default=40e6)
    ap.add_argument("--mass-amu", type=float, default=40.0)
    ap.add_argument("--charge-e", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=2e-6)
    ap.add_argument("--depth-ray-length", type=float, default=200e-6)
    ap.add_argument("--depth-nrays", type=int, default=48)
    ap.add_argument("--prefix", type=str, default="auto")
    ap.add_argument("--vrf", type=float, default=1.0)
    ap.add_argument("--coord-unit", type=float, default=None)
    ap.add_argument("--no-depth", action="store_true")

    # RF-null search bounds passed through to run_case.py
    ap.add_argument("--r0-z-min", type=float, default=None,
                    help="Lower z bound for RF-null search (mesh units).")
    ap.add_argument("--r0-z-max", type=float, default=None,
                    help="Upper z bound for RF-null search (mesh units). "
                         "Auto-detected by run_case.py from electrode top + margin if omitted.")
    ap.add_argument("--r0-x-min", type=float, default=None)
    ap.add_argument("--r0-x-max", type=float, default=None)
    ap.add_argument("--r0-y-min", type=float, default=None)
    ap.add_argument("--r0-y-max", type=float, default=None)
    ap.add_argument("--r0-search-margin", type=float, default=None,
                    help="Margin (metres) above electrode top for z auto-detect. "
                         "Passed to run_case.py only when explicitly set.")
    ap.add_argument("--r0-x-auto", action="store_true",
                    help="Pass --r0-x-auto to run_case.py (auto-detect x bounds from RF geometry).")
    ap.add_argument("--outer-tags", type=int, nargs="*", default=[4],
                    help="Facet tags for the outer Neumann boundary (default: [4]). "
                         "Must match --outer-tags in run_case.py.")

    return ap



def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    param_specs = parse_param_specs(args.param)
    if not param_specs:
        raise ValueError("You must provide at least one --param range.")

    cfg = RunConfig(
        run_case_py=args.run_case,
        workdir=mkdir(args.workdir),
        mesh_template=args.mesh_template,
        rf_tags=list(args.rf_tags),
        ground_tags=list(args.ground_tags),
        basis_tags=list(args.basis_tags),
        degree=args.degree,
        rf_freq=args.rf_freq,
        mass_amu=args.mass_amu,
        charge_e=args.charge_e,
        h=args.h,
        depth_ray_length=args.depth_ray_length,
        depth_nrays=args.depth_nrays,
        prefix=args.prefix,
        vrf=args.vrf,
        coord_unit=args.coord_unit,
        no_depth=args.no_depth,
        r0_z_min=args.r0_z_min,
        r0_z_max=args.r0_z_max,
        r0_x_min=args.r0_x_min,
        r0_x_max=args.r0_x_max,
        r0_y_min=args.r0_y_min,
        r0_y_max=args.r0_y_max,
        r0_search_margin=args.r0_search_margin,
        r0_x_auto=args.r0_x_auto,
        outer_tags=list(args.outer_tags) if args.outer_tags else [],
    )

    config_dump = {
        "run_config": asdict(cfg),
        "param_specs": [asdict(p) for p in param_specs],
        "n_cases": args.n_cases,
        "seed": args.seed,
    }
    find_paths(config_dump)
    write_json(cfg.workdir / "automation_config.json", config_dump)

    results = run_parallel_random_search(
        cfg,
        param_specs,
        n_cases=args.n_cases,
        seed=args.seed,
        max_workers=args.max_workers,
        resume=args.resume,
    )
    print_best(results)


if __name__ == "__main__":
    main()