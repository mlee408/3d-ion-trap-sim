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

Example 1 — single-junction sweep (run from src/ directory)
-----------------------------------------------------------
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
  --workdir ./sweep_002 \
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

Example 2 — 2-junction sweep via assemble_mesh.py (preferred)
-------------------------------------------------------------
python automate.py \
  --run-case ./run_sweep_metrics.py \
  --mesh-template "python ../geometry/assemble_mesh.py \
    --rf ../geometry/rf.step \
    --dc ../geometry/dc.step \
    --ground ../geometry/ground.step \
    --njunctions 2 \
    --junction-pitch 0.600 \
    --lc-electrode {lc_electrode} \
    --lc-center {lc_center} \
    --lc-far {lc_far} \
    --pad-z-top {pad_z_top} \
    --nopopup \
    --out {mesh_path}" \
  --workdir ./sweep_2junc \
  --rf-tags 1 --ground-tags 3 --outer-tags 4 \
  --param lc_electrode:0.002:0.006 \
  --param lc_center:0.003:0.008 \
  --param lc_far:0.020:0.050 \
  --param pad_z_top:0.400:0.800 \
  --degree 2 --mass-amu 40.0 --rf-freq 40e6 --vrf 150 \
  --coord-unit 1e-3 --r0-x-auto --n-cases 20 --seed 42

Example 3 — 2-junction sweep via junction_assemble_gmsh.py
----------------------------------------------------------
Pre-assemble STEP geometry, then mesh with assemble_mesh.py in one
template command. Use this when you need junction_assemble_gmsh.py's
fragment-based assembly (e.g., custom STEP inputs or junction spacing).

python automate.py \
  --run-case ./run_sweep_metrics.py \
  --mesh-template "python ../geometry/junction_assemble_gmsh.py \
    --rf ../geometry/rf.step \
    --dc ../geometry/dc.step \
    --ground ../geometry/ground.step \
    --out {case_dir}/combined.step \
    --spacing {junction_spacing} \
    --no-brep --quiet \
    && python ../geometry/assemble_mesh.py \
    --rf {case_dir}/combined.step \
    --dc /dev/null --ground /dev/null \
    --lc-electrode {lc_electrode} \
    --lc-center {lc_center} \
    --lc-far {lc_far} \
    --nopopup \
    --out {mesh_path}" \
  --workdir ./sweep_2junc_assembled \
  --rf-tags 1 --ground-tags 3 --outer-tags 4 \
  --param junction_spacing:0.500:0.700 \
  --param lc_electrode:0.002:0.006 \
  --param lc_center:0.003:0.008 \
  --param lc_far:0.020:0.050 \
  --degree 2 --mass-amu 40.0 --rf-freq 40e6 --vrf 150 \
  --coord-unit 1e-3 --r0-x-auto --n-cases 20 --seed 42

Note: Example 2 (assemble_mesh.py --njunctions 2) is simpler and
preferred for most sweeps. Example 3 is for cases where you need
junction_assemble_gmsh.py's separate STEP assembly step.
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
    # Fast-metrics / skip flags forwarded to run_sweep_metrics.py
    fast_metrics: bool
    refine_rounds: Optional[int]
    skip_depth_y: bool
    skip_transport_scan: bool


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
    physical_min_ok: Optional[bool]
    hessian_status: Optional[str]
    grad_rel_at_r0: Optional[float]
    score_breakdown: Optional[Dict[str, Any]]
    report_path: Optional[str]
    stderr_path: Optional[str]
    stdout_path: Optional[str]
    error_message: Optional[str]
    elapsed_s: float
    rejection_reason: Optional[str] = None
    paper_comparison: Optional[Dict[str, Any]] = None


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
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Corrupt or partial JSON in {path} "
            f"(size={len(text)} bytes, first 200 chars: {text[:200]!r}): {e}"
        ) from e

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


def extract_metrics_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the subset of metrics that the scorer needs.

    Handles both report formats:
      - run_case.py      : keys 'depth.depth_eV', 'secular.freq_hz', 'r0_SI_m'
      - run_sweep_metrics.py : keys 'radial_depth_core_eV', 'strong_freq_min_hz',
                               'strong_freq_max_hz', 'r0_x_m'/'r0_y_m'/'r0_z_m'

    Validity fields (run_sweep_metrics.py only):
      physical_min_ok  — bool: r0.z > 0, r0.z < 0.85*z_max, hessian valid/borderline
      hessian_status   — str: "valid" | "borderline_numeric" | None
      grad_rel_at_r0   — float: |∇Ψ|/(|H|_F × h); < 0.1 near-stationary

    center_offset_m is the lateral (x, y) displacement only — z (trap height) is
    intentionally excluded because trap height is a geometric consequence of electrode
    dimensions and must not dominate the confinement score.
    """
    depth_eV = None
    min_freq_hz = None
    max_freq_hz = None
    mode_spread_hz = None
    center_offset_m = None
    physical_min_ok = None
    hessian_status = None
    grad_rel_at_r0 = None

    # ── run_sweep_metrics.py format (flat keys) ───────────────────────────
    if "radial_depth_core_eV" in report or "strong_freq_min_hz" in report:
        depth_eV    = safe_float(report.get("radial_depth_core_eV"))
        min_freq_hz = safe_float(report.get("strong_freq_min_hz"))
        max_freq_hz = safe_float(report.get("strong_freq_max_hz"))
        if min_freq_hz is not None and max_freq_hz is not None:
            mode_spread_hz = max_freq_hz - min_freq_hz
        # lateral offset only — z excluded to avoid penalising trap height
        x = safe_float(report.get("r0_x_m"))
        y = safe_float(report.get("r0_y_m"))
        lateral = [v for v in (x, y) if v is not None]
        if lateral:
            center_offset_m = math.sqrt(sum(v * v for v in lateral))
        # validity fields
        _pmo = report.get("physical_min_ok")
        physical_min_ok = bool(_pmo) if _pmo is not None else None
        hessian_status  = report.get("hessian_status")
        grad_rel_at_r0  = safe_float(report.get("grad_rel_at_r0"))
        return {
            "depth_eV": depth_eV,
            "min_freq_hz": min_freq_hz,
            "max_freq_hz": max_freq_hz,
            "mode_spread_hz": mode_spread_hz,
            "center_offset_m": center_offset_m,
            "physical_min_ok": physical_min_ok,
            "hessian_status": hessian_status,
            "grad_rel_at_r0": grad_rel_at_r0,
        }

    # ── run_case.py format (nested keys) ─────────────────────────────────
    depth = report.get("depth")
    if isinstance(depth, dict):
        depth_eV = safe_float(depth.get("depth_eV"))

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

    # lateral offset from first two SI components (x, y); exclude z (trap height)
    r0_si = report.get("r0_SI_m")
    if isinstance(r0_si, list) and len(r0_si) >= 2:
        lateral = [safe_float(v) for v in r0_si[:2]]
        lateral = [v for v in lateral if v is not None]
        if lateral:
            center_offset_m = math.sqrt(sum(v * v for v in lateral))

    return {
        "depth_eV": depth_eV,
        "min_freq_hz": min_freq_hz,
        "max_freq_hz": max_freq_hz,
        "mode_spread_hz": mode_spread_hz,
        "center_offset_m": center_offset_m,
        "physical_min_ok": None,   # not available from run_case.py format
        "hessian_status": None,
        "grad_rel_at_r0": None,
    }



def classify_rejection(metrics: Dict[str, Any], status: str,
                       error_message: Optional[str] = None) -> Optional[str]:
    """Return a short rejection-reason tag, or None if the case is scorable.

    Rejection reasons (mutually exclusive, checked in order):
      "subprocess_crash" — run_case.py / mesh gen returned non-zero or threw
      "corrupt_json"     — JSON report could not be parsed
      "no_report"        — JSON report file was not found
      "invalid_physical" — physical_min_ok == False (far-field / electrode minimum)
      "missing_metrics"  — depth_eV or min_freq_hz is None
      None               — case is scorable
    """
    if status == "failed":
        if error_message and "Could not locate" in error_message:
            return "no_report"
        if error_message and ("JSON" in error_message or "json" in error_message.lower()):
            return "corrupt_json"
        return "subprocess_crash"
    if metrics.get("physical_min_ok") is False:
        return "invalid_physical"
    if metrics.get("depth_eV") is None or metrics.get("min_freq_hz") is None:
        return "missing_metrics"
    return None


def score_case_metrics(metrics: Dict[str, Any]) -> Optional[float]:
    """Scalar objective for local RF confinement quality.

    Hard rejects (return None — treated as failures by the Bayesian surrogate):
      - physical_min_ok == False  (far-field or electrode-adjacent minimum)
      - depth_eV or min_freq_hz missing/NaN

    Score terms (all ~O(1–5) for typical Ca+ geometries at 40 MHz, 150 V):
      +5.0    × depth_eV          radial_depth_core_eV — local confinement depth
      +1e-6   × min_freq_hz       weaker of top-2 radial modes (axial excluded)
      −0.3e-6 × spread_excess_hz  excess radial mode spread beyond 5 MHz target band

    Mode-spread policy:
      Spread below SPREAD_TARGET_HZ (5 MHz) is free — some asymmetry is expected
      in non-square window patterns (e.g. n=3).  Only the excess is penalised, and
      at 0.3e-6 the penalty grows slowly enough that a 10 MHz excess costs ~3 points
      (comparable to the depth term) rather than swamping it.  Observed n=3 sweeps
      had ~16 MHz spread, which would have cost 16 under the old raw-spread policy
      but costs only ~3.3 here, keeping the score meaningful.

    Soft multiplier (score × 0.85) for borderline_numeric Hessian:
      Applied after all additive terms so it scales the full score down uniformly.
      Ensures numerically uncertain cases rank below clearly valid ones with similar
      depth/frequency, without hard-rejecting them entirely.
    """
    # ── Hard reject ──────────────────────────────────────────────────────────
    if metrics.get("physical_min_ok") is False:
        return None

    depth_eV       = metrics.get("depth_eV")
    min_freq_hz    = metrics.get("min_freq_hz")
    mode_spread_hz = metrics.get("mode_spread_hz")

    if depth_eV is None or min_freq_hz is None:
        return None

    # ── Additive terms ───────────────────────────────────────────────────────
    term_depth = 5.0  * depth_eV
    term_freq  = 1e-6 * min_freq_hz

    # Excess-only spread penalty: free up to 5 MHz, then 0.3 pts per MHz beyond.
    _SPREAD_TARGET_HZ  = 5e6
    _SPREAD_COEFF      = 0.3e-6
    spread_excess_hz   = max(0.0, mode_spread_hz) - _SPREAD_TARGET_HZ if mode_spread_hz is not None else 0.0
    term_spread_pen    = _SPREAD_COEFF * max(0.0, spread_excess_hz)

    score = term_depth + term_freq - term_spread_pen

    # ── Soft multiplier for borderline Hessian ───────────────────────────────
    if metrics.get("hessian_status") == "borderline_numeric":
        score *= 0.85

    # Store breakdown for the caller to log (does not affect the score value).
    metrics["_score_breakdown"] = {
        "term_depth": round(term_depth, 4),
        "term_freq": round(term_freq, 4),
        "term_spread_pen": round(-term_spread_pen, 4),
        "spread_excess_mhz": round(max(0.0, spread_excess_hz) / 1e6, 3),
    }

    return score


# -----------------------------------------------------------------------------
# Paper benchmark comparison
# -----------------------------------------------------------------------------

# Reference values from published 3D-printed micro ion trap papers.
#
# Sources:
#   Paper 1: "3D-Printed Micro Ion Trap Technology for Scalable Quantum
#             Information Processing"
#     - Ion: Ca-40
#     - RF: 51.6 MHz (experiment); comparison simulation: 80 MHz, 150 V
#     - Trap geometry: opposing RF electrode separation 200 µm,
#       ion-to-RF distance 100 µm, RF pillar height 300 µm
#     - Experimental radial frequency range: 2.09–24.15 MHz
#
#   Paper 2: "Feasibility Study of 3D-Printed Micro Junction Array for
#             Ion Trap Quantum Processor"
#     - Ion: Yb-171  (mass 171 amu)
#     - RF: 44.3 MHz, 190 V
#     - Geometry: RF electrode width 120 µm, height 247 µm,
#       RF cross-section 56 µm × 82 µm, junction spacing 600 µm
#     - LINEAR REGION: trap height 82.3 µm, freq 2.32 MHz,
#       q_z = 0.15, depth 2.3 eV
#
# IMPORTANT: Paper 2 uses Yb-171 at 44.3 MHz / 190 V.  Our sweeps
# typically use Ca-40 at 40 MHz / ~150 V.  Geometry-only metrics
# (trap height) are directly comparable; frequency and depth scale
# with ion mass and operating conditions.  The comparison notes
# when operating conditions differ.
#
# Pseudopotential scaling:
#   Ψ ∝ q²|∇φ|² / (4 m ω²)  →  depth ∝ V_rf² / (m × f_rf²)
#   freq ∝ V_rf / (m × f_rf × coord_scale²)^(1/2)
# For Ca-40 at 40 MHz / 150 V vs Yb-171 at 44.3 MHz / 190 V
# on the same geometry, the Ca-40 case has ~3.3× deeper well
# and ~1.8× higher secular frequencies due to lighter mass.

PAPER_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    # ── Paper 2: Junction array linear region (primary reference) ────────
    # This is the main reference for our n=3/n=4 geometry sweeps.
    # Values are from the paper directly (Yb-171, 44.3 MHz, 190 V).
    "paper2_linear_yb171": {
        "description": "Paper 2 junction array, linear region (Yb-171, 44.3 MHz, 190 V)",
        "ion": "Yb-171",
        "mass_amu": 171.0,
        "rf_freq_mhz": 44.3,
        "vrf_V": 190.0,
        "trap_height_um": 82.3,
        "trap_height_tolerance_um": 15.0,
        "radial_freq_mhz": 2.32,       # single reported radial freq
        "q_z": 0.15,
        "depth_eV": 2.3,               # total trap depth (3D design)
        "depth_surface_trap_eV": 0.074, # comparison: surface trap = 74 meV
        "transport_barrier_eV": 0.007,  # pseudopotential barrier along CTC < 7 meV
        "junction_trap_height_um": 40.0,  # trap height near junction centre
        "notes": (
            "Paper values for Yb-171 at 44.3 MHz / 190 V.  For Ca-40 at "
            "40 MHz / 150 V on the same geometry, expect ~3.3× deeper well "
            "and ~1.8× higher secular frequencies due to lighter mass."
        ),
    },
    # ── Paper 2 scaled to Ca-40 / 40 MHz / 150 V (our typical sweep) ────
    # Approximate scaled values for direct comparison with our sweep output.
    # Trap height is geometry-only (no scaling needed).
    # Freq scales as: f_Ca ≈ f_Yb × sqrt(m_Yb/m_Ca) × (V_Ca/V_Yb) / (f_rf_Ca/f_rf_Yb)
    #   = 2.32 × sqrt(171/40) × (150/190) × (44.3/40) ≈ 2.32 × 2.07 × 0.789 × 1.108 ≈ 4.2 MHz
    # Depth scales as: D_Ca ≈ D_Yb × (V_Ca/V_Yb)² × (m_Yb/m_Ca) × (f_rf_Yb/f_rf_Ca)²
    #   = 2.3 × 0.623 × 4.275 × 1.226 ≈ 7.5 eV  (but capped by finite mesh domain)
    "paper2_linear_ca40_scaled": {
        "description": "Paper 2 linear region, scaled to Ca-40, 40 MHz, 150 V (approximate)",
        "ion": "Ca-40",
        "mass_amu": 40.0,
        "rf_freq_mhz": 40.0,
        "vrf_V": 150.0,
        "trap_height_um": 82.3,
        "trap_height_tolerance_um": 15.0,
        "strong_freq_min_mhz": 4.0,    # approximate scaled radial frequency
        "strong_freq_max_mhz": None,    # only one radial freq reported in paper
        "depth_z_eV": None,             # not directly available from scaling
        "radial_depth_core_eV": None,   # not directly available from scaling
        "depth_eV_approximate": 7.5,    # very approximate; depends on mesh extent
        "notes": (
            "Approximately scaled from Paper 2 Yb-171 values.  Trap height is "
            "geometry-only and directly comparable.  Frequency and depth scalings "
            "are approximate — actual values depend on mesh resolution, boundary "
            "conditions, and the exact electrode geometry used in the sweep."
        ),
    },
    # ── Paper 1: Single-junction 3D-printed trap (Ca-40) ─────────────────
    "paper1_ca40_experiment": {
        "description": "Paper 1 3D-printed single trap (Ca-40, 51.6 MHz experimental)",
        "ion": "Ca-40",
        "mass_amu": 40.0,
        "rf_freq_mhz": 51.6,
        "vrf_V": 160.0,     # max explored experimentally
        "ion_height_above_dc_um": 130.0,
        "rf_pillar_height_um": 300.0,
        "opposing_rf_separation_um": 200.0,
        "ion_to_rf_distance_um": 100.0,
        "radial_freq_range_mhz": [2.09, 24.15],
        "highest_radial_freq_mhz": 24.15,
        "q_at_highest": 0.903,
        "notes": (
            "Paper 1 experimental values.  Different geometry (single junction, "
            "taller RF pillars, wider RF separation) — not directly comparable "
            "to the 5-wire junction array geometry."
        ),
    },
}


def compare_to_paper_benchmarks(
    report: Dict[str, Any],
    metrics: Dict[str, Any],
    *,
    benchmark_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Compare case metrics to paper benchmark values.

    Returns a dict with per-metric signed differences, relative differences,
    per-metric verdicts, and an overall verdict label.  Returns None if
    insufficient metrics are available for any meaningful comparison.

    Default benchmark: "paper2_linear_ca40_scaled" (Paper 2 junction array,
    linear region, approximately scaled to Ca-40 / 40 MHz / 150 V).

    For raw paper values (Yb-171), use benchmark_key="paper2_linear_yb171".

    Parameters
    ----------
    report        : full JSON report from run_case.py or run_sweep_metrics.py
    metrics       : extracted metrics dict from extract_metrics_from_report()
    benchmark_key : explicit benchmark key from PAPER_BENCHMARKS
    """
    # ── Select benchmark ─────────────────────────────────────────────────────
    if benchmark_key and benchmark_key in PAPER_BENCHMARKS:
        bench = PAPER_BENCHMARKS[benchmark_key]
    else:
        # Default: use Ca-40 scaled version of Paper 2 linear region.
        bench = PAPER_BENCHMARKS["paper2_linear_ca40_scaled"]
    bench_key = benchmark_key or "paper2_linear_ca40_scaled"

    # Check whether case and benchmark operating conditions match
    _case_ion_mass = safe_float(report.get("mass_amu"))
    _case_rf_freq = safe_float(report.get("rf_freq_Hz"))
    _case_vrf = safe_float(report.get("vrf_V"))
    _bench_mass = bench.get("mass_amu")
    _bench_rf = bench.get("rf_freq_mhz")
    _bench_vrf = bench.get("vrf_V")
    operating_match = True
    mismatch_notes: List[str] = []
    if _case_ion_mass and _bench_mass and abs(_case_ion_mass - _bench_mass) > 0.5:
        operating_match = False
        mismatch_notes.append(f"ion mass: case={_case_ion_mass} vs bench={_bench_mass} amu")
    if _case_rf_freq and _bench_rf:
        _case_rf_mhz = _case_rf_freq / 1e6
        if abs(_case_rf_mhz - _bench_rf) / _bench_rf > 0.05:
            operating_match = False
            mismatch_notes.append(f"RF freq: case={_case_rf_mhz:.1f} vs bench={_bench_rf} MHz")
    if _case_vrf and _bench_vrf and abs(_case_vrf - _bench_vrf) / _bench_vrf > 0.1:
        operating_match = False
        mismatch_notes.append(f"V_RF: case={_case_vrf} vs bench={_bench_vrf} V")

    # ── Gather case values ───────────────────────────────────────────────────
    r0_z_m = None
    if "r0_z_m" in report:
        r0_z_m = safe_float(report["r0_z_m"])
    elif isinstance(report.get("r0_SI_m"), list) and len(report["r0_SI_m"]) >= 3:
        r0_z_m = safe_float(report["r0_SI_m"][2])
    trap_height_um = r0_z_m * 1e6 if r0_z_m is not None else None

    min_freq_mhz = metrics["min_freq_hz"] / 1e6 if metrics.get("min_freq_hz") else None
    max_freq_mhz = metrics["max_freq_hz"] / 1e6 if metrics.get("max_freq_hz") else None

    depth_z_eV = safe_float(report.get("depth_z_eV"))
    radial_depth_core_eV = safe_float(report.get("radial_depth_core_eV"))
    if radial_depth_core_eV is None:
        radial_depth_core_eV = metrics.get("depth_eV")

    # Also check for transport barrier if available
    transport_barrier_eV = safe_float(report.get("transport_barrier_xscan_eV"))

    # ── Per-metric comparison ────────────────────────────────────────────────
    comparisons: List[Dict[str, Any]] = []

    def _compare(name: str, case_val: Optional[float], paper_val: Optional[float],
                 higher_is_better: bool = True, unit: str = "",
                 tolerance_frac: float = 0.0) -> None:
        if case_val is None or paper_val is None:
            return
        diff = case_val - paper_val
        rel_diff = diff / abs(paper_val) if abs(paper_val) > 1e-30 else float("inf")
        if abs(rel_diff) <= tolerance_frac:
            verdict = "comparable"
        elif (diff > 0) == higher_is_better:
            verdict = "better"
        else:
            verdict = "worse"
        comparisons.append({
            "metric": name,
            "case_value": round(case_val, 6),
            "paper_value": round(paper_val, 6),
            "difference": round(diff, 6),
            "relative_diff_pct": round(rel_diff * 100, 1),
            "verdict": verdict,
            "unit": unit,
        })

    # Trap height — geometry-only, always directly comparable
    if trap_height_um is not None and bench.get("trap_height_um") is not None:
        _paper_h = bench["trap_height_um"]
        _tol = bench.get("trap_height_tolerance_um", 15.0)
        _diff = trap_height_um - _paper_h
        _rel = _diff / _paper_h if _paper_h > 0 else 0
        if abs(_diff) <= _tol:
            _h_verdict = "comparable"
        else:
            _h_verdict = "higher" if _diff > 0 else "lower"
        comparisons.append({
            "metric": "trap_height",
            "case_value": round(trap_height_um, 2),
            "paper_value": round(_paper_h, 2),
            "difference": round(_diff, 2),
            "relative_diff_pct": round(_rel * 100, 1),
            "verdict": _h_verdict,
            "unit": "µm",
        })

    # Secular frequencies — the benchmark may have a single "radial_freq_mhz"
    # (Paper 2 reports one frequency) or separate strong_freq_min/max.
    _bench_freq = bench.get("strong_freq_min_mhz") or bench.get("radial_freq_mhz")
    _compare("strong_freq_min", min_freq_mhz, _bench_freq,
             higher_is_better=True, unit="MHz", tolerance_frac=0.10)
    _compare("strong_freq_max", max_freq_mhz, bench.get("strong_freq_max_mhz"),
             higher_is_better=True, unit="MHz", tolerance_frac=0.10)

    # Depths — higher is better
    _bench_depth = bench.get("depth_z_eV") or bench.get("depth_eV")
    _compare("depth_z", depth_z_eV, _bench_depth,
             higher_is_better=True, unit="eV", tolerance_frac=0.15)
    _compare("radial_depth_core", radial_depth_core_eV, bench.get("radial_depth_core_eV"),
             higher_is_better=True, unit="eV", tolerance_frac=0.15)

    # Transport barrier — lower is better (easier ion shuttling)
    _bench_transport = bench.get("transport_barrier_eV")
    if transport_barrier_eV is not None and _bench_transport is not None:
        _compare("transport_barrier", transport_barrier_eV, _bench_transport,
                 higher_is_better=False, unit="eV", tolerance_frac=0.20)

    if not comparisons:
        return None

    # ── Overall verdict ──────────────────────────────────────────────────────
    confinement_metrics = [c for c in comparisons if c["metric"] != "trap_height"]
    conf_verdicts = [c["verdict"] for c in confinement_metrics]
    conf_better = conf_verdicts.count("better")
    conf_worse = conf_verdicts.count("worse")

    if conf_worse == 0 and conf_better > 0:
        overall = "better"
    elif conf_better == 0 and conf_worse > 0:
        overall = "worse"
    elif conf_better > 0 and conf_worse > 0:
        better_names = [c["metric"] for c in confinement_metrics if c["verdict"] == "better"]
        worse_names = [c["metric"] for c in confinement_metrics if c["verdict"] == "worse"]
        overall = f"mixed: better on {','.join(better_names)}; worse on {','.join(worse_names)}"
    else:
        overall = "comparable"

    if not operating_match:
        overall += f" (operating conditions differ: {'; '.join(mismatch_notes)})"

    # Summary one-liner for logs
    parts = []
    for c in comparisons:
        sign = "+" if c["difference"] >= 0 else ""
        parts.append(f"{c['metric']}={c['case_value']}{c['unit']} "
                     f"({sign}{c['relative_diff_pct']}% vs paper)")
    summary_line = "; ".join(parts)

    return {
        "benchmark_used": bench.get("description", "unknown"),
        "benchmark_key": bench_key,
        "operating_conditions_match": operating_match,
        "operating_mismatch": mismatch_notes if mismatch_notes else None,
        "comparisons": comparisons,
        "verdict": overall,
        "summary": summary_line,
        "caveat": bench.get("notes", ""),
    }


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
    - all parameter names, e.g. {rf_height}, {junction_spacing}
    - {case_dir}  — resolved absolute path to the case working directory
    - {mesh_path} — resolved absolute path to the expected output mesh file

    Multi-step commands:
      If the template contains shell operators (&&, ||, ;, |) the command is
      executed via ``sh -c "..."`` instead of direct exec.  This allows
      chained pipelines such as::

          python junction_assemble_gmsh.py ... --out {case_dir}/combined.step && \
          python assemble_mesh.py --rf {case_dir}/combined.step ... --out {mesh_path}
    """
    mesh_path = (case_dir / "mesh.msh").resolve()
    mapping: Dict[str, Any] = dict(params)
    mapping["case_dir"] = str(case_dir.resolve())
    mapping["mesh_path"] = str(mesh_path)

    command_str = render_template(mesh_template, mapping)

    # Detect shell operators that require sh -c invocation
    _shell_ops = ("&&", "||", "|", ";")
    needs_shell = any(op in command_str for op in _shell_ops)

    if needs_shell:
        cmd: Any = command_str   # will be passed as string with shell=True
    else:
        cmd = shlex.split(command_str)

    gen_stdout = case_dir / "meshgen.stdout.txt"
    gen_stderr = case_dir / "meshgen.stderr.txt"
    with gen_stdout.open("w") as fout, gen_stderr.open("w") as ferr:
        proc = subprocess.run(
            cmd,
            cwd=None,
            stdout=fout,
            stderr=ferr,
            text=True,
            shell=needs_shell,
        )
    rc = int(proc.returncode)

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
    # Detect whether we are calling run_sweep_metrics.py (lightweight) or
    # run_case.py (full). run_sweep_metrics.py does not accept --h,
    # --depth-ray-length, or --no-depth.
    is_sweep = cfg.run_case_py.name == "run_sweep_metrics.py"

    cmd: List[str] = [
        sys.executable,
        str(cfg.run_case_py.resolve()),
        "--mesh", str(mesh_path.resolve()),
        "--outdir", str(case_dir.resolve()),
        "--degree", str(cfg.degree),
        "--rf-freq", str(cfg.rf_freq),
        "--mass-amu", str(cfg.mass_amu),
        "--charge-e", str(cfg.charge_e),
        "--depth-nrays", str(cfg.depth_nrays),
        "--prefix", case_prefix,
        "--vrf", str(cfg.vrf),
    ]

    # run_case.py-only arguments
    if not is_sweep:
        cmd.extend(["--h", str(cfg.h)])
        cmd.extend(["--depth-ray-length", str(cfg.depth_ray_length)])
        if cfg.no_depth:
            cmd.append("--no-depth")

    if cfg.coord_unit is not None:
        cmd.extend(["--coord-unit", str(cfg.coord_unit)])

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

    # Fast-metrics / skip flags (run_sweep_metrics.py only)
    if is_sweep:
        if cfg.fast_metrics:
            cmd.append("--fast-metrics")
        if cfg.refine_rounds is not None:
            cmd.extend(["--refine-rounds", str(cfg.refine_rounds)])
        if cfg.skip_depth_y:
            cmd.append("--skip-depth-y")
        if cfg.skip_transport_scan:
            cmd.append("--skip-transport-scan")

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
        case_dir / f"{case_prefix}_sweep.json",    # run_sweep_metrics.py
        case_dir / f"{case_prefix}_report.json",   # run_case.py
        case_dir / "report.json",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Fallback: search for a single json file in case_dir, excluding params.json
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
        rejection = classify_rejection(metrics, "ok")
        paper_cmp = compare_to_paper_benchmarks(report, metrics)

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
            physical_min_ok=metrics.get("physical_min_ok"),
            hessian_status=metrics.get("hessian_status"),
            grad_rel_at_r0=metrics.get("grad_rel_at_r0"),
            score_breakdown=metrics.get("_score_breakdown"),
            report_path=str(report_path),
            stderr_path=str(stderr_path),
            stdout_path=str(stdout_path),
            error_message=None,
            elapsed_s=elapsed,
            rejection_reason=rejection,
            paper_comparison=paper_cmp,
        )
    except Exception as e:
        elapsed = time.time() - start
        err_msg = str(e)
        rejection = classify_rejection({}, "failed", error_message=err_msg)
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
            physical_min_ok=None,
            hessian_status=None,
            grad_rel_at_r0=None,
            score_breakdown=None,
            report_path=None,
            stderr_path=str(stderr_path),
            stdout_path=str(stdout_path),
            error_message=err_msg,
            elapsed_s=elapsed,
            rejection_reason=rejection,
        )


# -----------------------------------------------------------------------------
# Search loop
# -----------------------------------------------------------------------------


def _result_row(result: CaseResult) -> Dict[str, Any]:
    row = {
        "case_id": result.case_id,
        "status": result.status,
        "score": result.score,
        "rejection_reason": result.rejection_reason,
        "depth_eV": result.depth_eV,
        "min_freq_hz": result.min_freq_hz,
        "max_freq_hz": result.max_freq_hz,
        "mode_spread_hz": result.mode_spread_hz,
        "center_offset_m": result.center_offset_m,
        "physical_min_ok": result.physical_min_ok,
        "hessian_status": result.hessian_status,
        "grad_rel_at_r0": result.grad_rel_at_r0,
        "mesh_path": result.mesh_path,
        "case_dir": result.case_dir,
        "report_path": result.report_path,
        "stderr_path": result.stderr_path,
        "stdout_path": result.stdout_path,
        "error_message": result.error_message,
        "elapsed_s": result.elapsed_s,
        **{f"param_{k}": v for k, v in result.params.items()},
    }
    # Include paper comparison fields inline if available
    if result.paper_comparison:
        row["paper_verdict"] = result.paper_comparison.get("verdict")
        row["paper_summary"] = result.paper_comparison.get("summary")
    return row


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


def _params_to_list(params: Dict[str, float], specs: Sequence[ParamSpec]) -> List[float]:
    return [params[s.name] for s in specs]


def _list_to_params(values: List[float], specs: Sequence[ParamSpec]) -> Dict[str, float]:
    return {s.name: float(v) for s, v in zip(specs, values)}


def run_bayesian_search(
    cfg: RunConfig,
    param_specs: Sequence[ParamSpec],
    *,
    n_cases: int,
    n_random_start: int,
    seed: int,
    max_workers: int,
    resume: bool,
) -> List[CaseResult]:
    """Bayesian optimisation loop using scikit-optimize's Gaussian Process surrogate.

    Strategy
    --------
    1. Run ``n_random_start`` cases with Latin-hypercube sampling to explore the
       space before the GP has enough data to be useful.
    2. After each completed batch, fit a GP surrogate to all scored results so far
       and use Expected Improvement (EI) acquisition to pick the next point(s).
    3. Failed cases (score=None) are assigned a penalty score so the surrogate
       learns to avoid those regions of parameter space.
    4. Progress is written to summary.csv / summary.jsonl after every completed
       case, exactly as in the random search, so runs are resumable.

    Falls back to pure random search if scikit-optimize is not installed.
    """
    try:
        from skopt import Optimizer
        from skopt.space import Real
        _has_skopt = True
    except ImportError:
        _has_skopt = False
        print("[bayes] scikit-optimize not found — falling back to random search.")
        print("[bayes] Install with: pip install scikit-optimize")

    from concurrent.futures import ProcessPoolExecutor, as_completed

    rng = random.Random(seed)
    results: List[CaseResult] = []
    summary_csv = cfg.workdir / "summary.csv"
    all_jsonl   = cfg.workdir / "summary.jsonl"

    # ── Resume: reload already-completed cases ────────────────────────────
    existing = _load_existing_results(summary_csv) if resume else {}
    case_counter = [max((int(k.split("_")[1]) for k in existing), default=-1) + 1]

    def next_case_index() -> int:
        idx = case_counter[0]
        case_counter[0] += 1
        return idx

    # Track rejection reasons across the sweep for health summaries
    rejection_counts: Dict[str, int] = {}

    def record(result: CaseResult) -> None:
        results.append(result)
        row = _result_row(result)
        append_csv_row(summary_csv, row)
        with all_jsonl.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        # Track rejection reasons
        if result.rejection_reason:
            rejection_counts[result.rejection_reason] = (
                rejection_counts.get(result.rejection_reason, 0) + 1
            )
        _bd = result.score_breakdown or {}
        _bd_str = ""
        if _bd:
            _bd_str = (
                f"  depth={_bd.get('term_depth', 0):.3g}"
                f"  freq={_bd.get('term_freq', 0):.3g}"
                f"  spread_pen={_bd.get('term_spread_pen', 0):.3g}"
                f"  (excess={_bd.get('spread_excess_mhz', 0):.2g} MHz)"
            )
        _rej_str = f"  REJECT={result.rejection_reason}" if result.rejection_reason else ""
        _paper_str = ""
        if result.paper_comparison and result.paper_comparison.get("verdict"):
            _paper_str = f"  paper={result.paper_comparison['verdict']}"
        print(
            f"[{result.case_id}] {result.status} | score={result.score} "
            f"| hessian={result.hessian_status} phys_ok={result.physical_min_ok}"
            f"{_bd_str}{_rej_str}{_paper_str}"
            f" | elapsed={result.elapsed_s:.1f}s"
        )

    def submit_batch(executor, param_list: List[Dict[str, float]]) -> List[CaseResult]:
        futures = {
            executor.submit(evaluate_case, next_case_index(), p, cfg=cfg): p
            for p in param_list
        }
        batch_results = []
        for fut in as_completed(futures):
            r = fut.result()
            record(r)
            batch_results.append(r)
        return batch_results

    # Penalty used for failed cases so the surrogate avoids those regions.
    # Set to slightly below the worst observed score, updated as runs come in.
    def failure_penalty(scored: List[CaseResult]) -> float:
        valid_scores = [r.score for r in scored if r.score is not None]
        if not valid_scores:
            return -1e6
        return min(valid_scores) - abs(min(valid_scores)) * 0.1

    # ── Seed the optimizer with existing results if resuming ─────────────
    all_x: List[List[float]] = []
    all_y: List[float] = []
    if resume and existing:
        print(f"[bayes] Resume: loading {len(existing)} existing rows.")
        for row in existing.values():
            try:
                x = [float(row[f"param_{s.name}"]) for s in param_specs]
                y_raw = row.get("score", "")
                y = float(y_raw) if y_raw not in ("", "None", None) else None
                all_x.append(x)
                all_y.append(y)  # type: ignore[arg-type]
            except (KeyError, ValueError):
                pass

    if not _has_skopt:
        # ── Pure random fallback ──────────────────────────────────────────
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            remaining = n_cases - len(existing)
            param_list = [sample_random_params(param_specs, rng) for _ in range(remaining)]
            submit_batch(ex, param_list)
        return results

    # ── Bayesian loop ─────────────────────────────────────────────────────
    dimensions = [Real(s.low, s.high, name=s.name) for s in param_specs]
    opt = Optimizer(
        dimensions=dimensions,
        base_estimator="GP",
        acq_func="EI",           # Expected Improvement
        acq_optimizer="lbfgs",
        n_initial_points=n_random_start,
        random_state=seed,
    )

    # Feed existing results into the optimizer so it starts warm.
    if all_x:
        # Compute penalty from existing scores (results list is empty during
        # resume — use the loaded all_y values instead).
        _valid_existing = [y for y in all_y if y is not None]
        if _valid_existing:
            pen_warm = min(_valid_existing) - abs(min(_valid_existing)) * 0.1
        else:
            pen_warm = -1e6
        y_feed = [(-y if y is not None else -pen_warm) for y in all_y]
        # skopt minimizes, so negate the score (we want to maximize).
        opt.tell(all_x, y_feed)
        print(f"[bayes] Warm-started optimizer with {len(all_x)} existing points.")

    total_done = len(existing)
    remaining  = n_cases - total_done

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        while remaining > 0:
            # Ask for a batch of suggestions (one per worker, up to remaining).
            batch_size = min(max_workers, remaining)

            # During initial random exploration, ask() returns LHS points.
            # After n_initial_points, it uses the GP + EI acquisition.
            suggested = opt.ask(n_points=batch_size)
            if not isinstance(suggested[0], list):
                suggested = [suggested]   # single-point ask returns a flat list

            param_batch = [_list_to_params(x, param_specs) for x in suggested]
            batch_results = submit_batch(ex, param_batch)

            # Update the surrogate with completed results.
            pen = failure_penalty(results)
            xs_new = [_params_to_list(r.params, param_specs) for r in batch_results]
            ys_new = [-(r.score if r.score is not None else pen) for r in batch_results]
            opt.tell(xs_new, ys_new)

            remaining -= len(batch_results)
            total_done += len(batch_results)

            # Print current best + sweep health summary.
            valid = [r for r in results if r.score is not None]
            n_total = len(results)
            n_valid = len(valid)
            yield_pct = 100.0 * n_valid / n_total if n_total > 0 else 0.0
            if valid:
                best = max(valid, key=lambda r: r.score)  # type: ignore[arg-type]
                print(
                    f"[bayes] {total_done}/{total_done + remaining} done | "
                    f"best so far: {best.case_id} score={best.score:.4g} "
                    f"params={best.params}"
                )
            print(
                f"[sweep health] {n_valid}/{n_total} valid ({yield_pct:.0f}% yield)"
                + (f"  rejections: {rejection_counts}" if rejection_counts else "")
            )

    return results


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------



def print_best(results: Sequence[CaseResult], *, top_k: int = 5) -> None:
    valid = [r for r in results if r.status == "ok" and r.score is not None]
    valid.sort(key=lambda r: r.score if r.score is not None else -1e99, reverse=True)

    # ── Sweep health summary ─────────────────────────────────────────────────
    n_total = len(results)
    n_valid = len(valid)
    n_failed = sum(1 for r in results if r.status == "failed")
    n_rejected = sum(1 for r in results if r.status == "ok" and r.score is None)
    rej_reasons: Dict[str, int] = {}
    for r in results:
        if r.rejection_reason:
            rej_reasons[r.rejection_reason] = rej_reasons.get(r.rejection_reason, 0) + 1

    print("\nSweep health summary")
    print("====================")
    print(f"  Total cases:      {n_total}")
    print(f"  Valid (scored):    {n_valid}  ({100*n_valid/n_total:.0f}% yield)" if n_total else "")
    print(f"  Failed (crash):   {n_failed}")
    print(f"  Rejected (no score): {n_rejected}")
    if rej_reasons:
        print(f"  Rejection breakdown: {rej_reasons}")

    print("\nTop results")
    print("===========")
    if not valid:
        print("No successful cases with valid scores.")
        return

    for r in valid[:top_k]:
        _paper = ""
        if r.paper_comparison and r.paper_comparison.get("verdict"):
            _paper = f", paper={r.paper_comparison['verdict']}"
        print(
            f"{r.case_id}: score={r.score:.6g}, depth_eV={r.depth_eV}, "
            f"min_freq_hz={r.min_freq_hz}, mode_spread_hz={r.mode_spread_hz}"
            f"{_paper}, case_dir={r.case_dir}"
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
    ap.add_argument("--n-random-start", type=int, default=5,
                    help="Number of random (Latin-hypercube) cases to run before the "
                         "Bayesian GP surrogate takes over. Rule of thumb: ~2-3× the "
                         "number of parameters. Default: 5.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-workers", type=int, default=4,
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

    # Fast-metrics / skip flags (forwarded to run_sweep_metrics.py only)
    ap.add_argument("--fast-metrics", action="store_true",
                    help="Enable fast-metrics mode in run_sweep_metrics.py "
                         "(fewer rays, fewer refinement rounds, skip transport scan and depth-y).")
    ap.add_argument("--refine-rounds", type=int, default=None,
                    help="Override number of coordinate-descent refinement rounds "
                         "(run_sweep_metrics.py only; default: 8, or 4 in --fast-metrics mode).")
    ap.add_argument("--skip-depth-y", action="store_true",
                    help="Skip y-direction trap depth scan (run_sweep_metrics.py only).")
    ap.add_argument("--skip-transport-scan", action="store_true",
                    help="Skip transport barrier scan (run_sweep_metrics.py only).")

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
        fast_metrics=args.fast_metrics,
        refine_rounds=args.refine_rounds,
        skip_depth_y=args.skip_depth_y,
        skip_transport_scan=args.skip_transport_scan,
    )

    config_dump = {
        "run_config": asdict(cfg),
        "param_specs": [asdict(p) for p in param_specs],
        "n_cases": args.n_cases,
        "seed": args.seed,
    }
    write_json(cfg.workdir / "automation_config.json", config_dump)

    results = run_bayesian_search(
        cfg,
        param_specs,
        n_cases=args.n_cases,
        n_random_start=args.n_random_start,
        seed=args.seed,
        max_workers=args.max_workers,
        resume=args.resume,
    )
    print_best(results)


if __name__ == "__main__":
    main()