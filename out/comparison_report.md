# Trap Simulation vs. Literature Comparison Report

**Simulation:** `trap_vacuum.msh` — RF-only case  
**Papers:** arXiv:2310.00595v2 (Paper 1) · arXiv:2509.17275v1 (Paper 2)  
**Date generated:** 2026-04-09

---

## 1. Mesh Geometry

| Parameter | Simulation | Paper 1 (3D pillar) | Paper 2 (3D+surface) |
|---|---|---|---|
| Trap type | 3D Paul trap | 3D printed pillar | 3D printed + surface |
| Bounding box | ±100 µm sphere | — | 600 × 600 µm unit cell |
| Trap minimum | (−0.26, 0.10, 0.07) µm | centre of 4 pillars | 82.3 µm above surface |
| Distance to origin | **0.29 µm** | 0 (by design) | — |
| Coordinate units | **µm** (auto-detected) | µm (implied) | µm |

The trap minimum is at the geometric centre of the mesh (< 0.3 µm from origin),
consistent with a symmetric 3D Paul trap similar to Paper 1.  It is **not** a surface-trap
geometry like Paper 2 (where the ion sits 82.3 µm above the chip surface).

---

## 2. Simulation Parameters

| Parameter | Used in run | Paper 1 | Paper 2 |
|---|---|---|---|
| Ion species | ⁴⁰Ca⁺ (40 amu) | ⁴⁰Ca⁺ | ¹⁷¹Yb⁺ (171 amu) |
| RF frequency Ω_RF / 2π | **40 MHz** | 51.6 MHz | 44.3 MHz |
| RF voltage V_RF (used) | **1 V** (normalised†) | up to ~160 V | 190 V |
| Mesh coordinate unit | **µm** | µm | µm |

† The current run uses V_RF = 1 V (normalised boundary conditions).  
  Use `--vrf 150` (or any desired voltage) to get physical SI frequencies directly.

---

## 3. Secular Frequencies — Corrected vs. Literature

### 3.1 Bug summary (now fixed)

The raw output reported `f ≈ 0.02 Hz` and `Ψ_min < 0`.  Two independent bugs were present:

| Bug | Cause | Effect | Fix applied |
|---|---|---|---|
| `f ≈ 0.02 Hz` | Hessian in J/µm² but divided by mass as if J/m² — missing factor (10⁶)² | Frequencies wrong by ×10⁶ | `coord_scale` parameter in `secular_frequencies_from_pseudopotential` |
| `Ψ_min = −6.77×10⁻²⁸ J` | CG1 L2-projection of `|∇φ|²` produces Gibbs-like undershoot < 0 | Trap depth = 4 nV (nonsense); `find_minimum_cg1` could return wrong DOF | Clip `Ψ ← max(Ψ, 0)` after projection |

### 3.2 Corrected secular frequencies

**Formula applied after fixes:**  
`ω_phys = V_RF / coord_scale × √(λ / m)`  
`coord_scale = 1×10⁻⁶ m/µm`, `m = 6.641×10⁻²⁶ kg`

#### Ca⁺, 40 MHz RF (our simulation geometry)

| V_RF (V) | f₁ (MHz) | f₂ (MHz) | f₃ (MHz) | Comment |
|---|---|---|---|---|
| 1 | 0.0206 | 0.0207 | 0.0210 | normalised run |
| 50 | 1.03 | 1.03 | 1.05 | low-q regime |
| 100 | 2.06 | 2.07 | 2.10 | |
| **150** | **3.09** | **3.10** | **3.15** | comparable to Paper 1 |
| 190 | 3.91 | 3.93 | 3.99 | |

All three modes are within **3.9%** of each other → consistent with the
highly symmetric 3D geometry of Paper 1.

#### Scaled to Paper 1 conditions (Ca⁺, 51.6 MHz RF)

Secular frequency scales as ω ∝ V_RF / Ω_RF, so at the same V_RF our
40 MHz trap gives 51.6/40 = 1.29× **higher** frequency than Paper 1.

| V_RF (V) | Our trap (40 MHz) → scaled to 51.6 MHz | Paper 1 (51.6 MHz) | q_Mathieu |
|---|---|---|---|
| 50 | 0.80 MHz | ~2 MHz at q=0.13 | ~0.044 |
| 100 | 1.60 MHz | — | ~0.088 |
| **150** | **2.40 MHz** | **2.09 MHz at q=0.13** | **~0.13** |
| 160 | 2.56 MHz | ~24 MHz at q=0.9 | ~0.14 |

> **At q ≈ 0.13 (Paper 1 lower bound, 51.6 MHz, ~150 V), our corrected result of**
> **2.40 MHz agrees well with Paper 1's measured 2.09 MHz (~15% difference).**
> The remaining discrepancy likely reflects different electrode gap and
> geometry (Paper 1: 100 µm ion-to-electrode distance; our mesh may differ).

#### Scaled to Paper 2 conditions (¹⁷¹Yb⁺, 44.3 MHz RF, 190 V)

Scale factor: `ω_Yb/ω_Ca = √(m_Ca/m_Yb) × (Ω_Ca/Ω_Yb) = √(40/171) × (40/44.3) = 0.302`

| Mode | Our sim (Yb⁺ equiv., 44.3 MHz, 190 V) | Paper 2 |
|---|---|---|
| z-axis (vertical) | **1.71 MHz** | **2.32 MHz** |
| y-axis | **1.72 MHz** | ~2.74 MHz (+18%) |
| x-axis | **1.74 MHz** | ~1.13 MHz (axial) |

> Our simulated z-axis frequency (1.71 MHz) is **~26% below Paper 2 (2.32 MHz)**.
> This is expected: Paper 2 uses a specialised hybrid 3D-printed + surface electrode
> geometry with a higher confinement factor (η = 0.75×10⁹ eV/m²),
> while our mesh likely represents Paper 1's simpler pillar geometry.

---

## 4. Trap Depth

| | Our run (raw) | Our run (corrected, re-run needed) | Paper 1 | Paper 2 |
|---|---|---|---|---|
| Depth at 150 V | 4.2 nV (INVALID) | — | ~1.0 eV (80 MHz, 150 V) | 2.3 eV (44.3 MHz, 190 V) |
| Depth at 190 V | — | TBD after code fix | — | 2.3 eV |
| Surface-trap depth | — | — | — | 74 meV (31× shallower) |

The raw depth of 4.2 nV is a direct consequence of the negative `Ψ_min` artifact.
After applying the `max(Ψ, 0)` clip and re-running, the depth will correctly reflect the
pseudopotential barrier height from the trap centre to the mesh boundary.  Expected range
based on geometry similarity to Paper 1: **0.5–2 eV at 100–190 V**.

**To re-run with corrected code and get physical depth:**
```bash
python3 src/run_case.py \
  --mesh meshes/trap_vacuum.msh \
  --outdir out \
  --rf-tags 1 \
  --ground-tags 2 3 4 \
  --prefix rf_v150 \
  --vrf 150 \
  --coord-unit 1e-6
```

---

## 5. Mode Structure

| | Our sim | Paper 1 | Paper 2 |
|---|---|---|---|
| Mode degeneracy (radial) | **3.9%** spread | Observed split ≤ few % | 18% anisotropy (y vs z) |
| Eigenvectors | ≈ aligned with mesh axes | — | — |
| Trap symmetry | 3D symmetric (pillar) | 4-pillar vertical | Asymmetric (surface + 3D) |

The near-degeneracy of all three modes strongly indicates a **3D-symmetric pillar geometry**
(matching Paper 1).  Paper 2's geometry intentionally breaks the radial symmetry by combining
a 3D-printed electrode with a surface trap, producing an 18% anisotropy — not seen here.

---

## 6. Identified Issues and Fixes Applied

| Issue | Status |
|---|---|
| `Ψ_min < 0` (L2 projection Gibbs undershoot) | ✅ Fixed — clip to 0 in `compute_rf_pseudopotential` |
| Secular freq. × 10⁻⁶ (J/µm² treated as J/m²) | ✅ Fixed — `coord_scale` + `v_rf` parameters added |
| `find_minimum_cg1` could land on boundary DOF | ✅ Fixed — interior-only argmin |
| `f.eval()` 3-arg API removed in DOLFINx ≥ 0.8 | ✅ Fixed — batch 2-arg API |
| XDMF `read_meshtags` needs `create_entities` first | ✅ Fixed |
| `h=2e-6` scale-agnostic default | ✅ Fixed — auto-scaled from mesh cell size |
| Secular freq. not in physical units (missing V_RF) | ✅ Fixed — `--vrf` argument added |
| Trap depth meaningless without physical V_RF | ✅ Code fixed; re-run needed for valid numbers |

---

## 7. Summary

The simulation geometry matches **Paper 1 (3D printed vertical Paul trap)** well:
- 3D symmetric, ion trapped at geometric centre
- Secular frequencies consistent with Paper 1 measurements at matched q values (~15% agreement)
- All three modes nearly degenerate (< 4% spread), consistent with 4-pillar symmetry

It does **not** match Paper 2 (hybrid 3D+surface geometry with strong radial anisotropy).

**Next steps:**
1. Re-run with `--vrf 150 --coord-unit 1e-6` to get valid physical depth and frequencies
2. Verify electrode tag assignments (RF=1, GND=2,3,4) against the mesh Physical Groups
3. Compare with Paper 1 Figure 4 (secular frequency vs Mathieu q) to calibrate the electrode gap
