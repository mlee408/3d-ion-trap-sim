# trap_sim

FEM ion-trap simulator (FEniCSx / Gmsh).

## Quick orientation

| What you want | Where to look |
|---|---|
| Solver & physics code | `src/` |
| Geometry-generation scripts | `geometry/` |
| Generated CAD (STEP from surface junction generator) | `cad/generated/` |
| Baseline meshes | `meshes/base/` |
| Generated meshes (parametric sweep) | `meshes/generated/` |
| Simulation run outputs | `runs/` |
| Sweep manifests / JSON / CSV metadata | `manifests/` |
| Papers and reference notes | `refs/` |
| Driver / batch scripts | `scripts/` |

## Directory reference

```
trap_sim/
├─ src/                 # laplace.py, metrics.py, mesh_io.py, run_case.py,
│                       # run_sweep_metrics.py, automate.py
├─ geometry/            # generate_surface_junction.py, make_rf_step.py,
│                       # rf_cell_gen.py, assemble_mesh.py, split_rf.py
├─ cad/
│  ├─ base/             # rf.step (hand-made baseline)
│  ├─ generated/        # parametric STEP from surface junction generator
│  └─ archive/
├─ meshes/
│  ├─ base/             # trap_2j.msh, trap_vacuum_fine.msh, …
│  ├─ generated/        # meshes from parametric CAD sweeps
│  └─ archive/
├─ runs/
│  ├─ baseline/         # benchmark run outputs
│  └─ sweeps/           # one subfolder per sweep campaign
├─ manifests/           # trap_2j_ref_sweep.json, trap_2j_sweep.json, …
├─ logs/
│  ├─ geometry/
│  └─ solver/
├─ refs/
│  ├─ papers/
│  └─ notes/
├─ scripts/             # make_mesh.py, make_mesh_parametric.py, plot_field_slices.py
└─ tmp/                 # scratch — not tracked
```

## Common workflows

**Generate baseline mesh**
```bash
python scripts/make_mesh.py --out meshes/generated/trap.msh
```

**Generate surface junction electrodes**
```bash
python geometry/make_rf_step.py --out-dir cad/generated/surface_junction
```

**Generate with 3D RF beams**
```bash
python geometry/make_rf_step.py --with-3d-rf --rf-height 267 --out-dir cad/generated/combined
```

**Run a single case**
```bash
python src/run_case.py \
    --mesh meshes/base/trap_2j.msh \
    --outdir runs/baseline/my_run/
```

**Run a sweep**
```bash
python src/automate.py \
    --run-case src/run_case.py \
    --mesh-template "python scripts/make_mesh_parametric.py ..." \
    --workdir runs/sweeps/rf_mesh_study/
```

## Environment

```bash
conda env create -f environment.yml
conda activate trap-sim
```
