#!/usr/bin/env python3
"""
merge_rf_with_base.py

Fuse an rf_cell_gen.py output STEP with rf_surface.step into a single
closed solid STEP that assemble_mesh.py can consume.

Usage:
    python merge_rf_with_base.py <rf_cell.step> <rf_surface.step> <output.step>
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: merge_rf_with_base.py <rf_cell.step> <rf_surface.step> <out.step>",
              file=sys.stderr)
        return 2

    rf_step   = Path(sys.argv[1]).resolve()
    base_step = Path(sys.argv[2]).resolve()
    out_step  = Path(sys.argv[3]).resolve()

    for p in (rf_step, base_step):
        if not p.exists():
            print(f"ERROR: missing input: {p}", file=sys.stderr)
            return 1

    import gmsh
    gmsh.initialize([sys.argv[0]])
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("merge")
    occ = gmsh.model.occ

    # ── Import RF cell ──
    before = set(t for _, t in occ.getEntities(3))
    occ.importShapes(str(rf_step))
    occ.synchronize()
    after = set(t for _, t in occ.getEntities(3))
    rf_tags = sorted(after - before)
    print(f"[merge] rf volumes: {len(rf_tags)} tags={rf_tags}")
    if not rf_tags:
        print(f"ERROR: {rf_step} imported 0 volumes", file=sys.stderr)
        return 1

    # ── Import base (rf_surface) ──
    before = set(t for _, t in occ.getEntities(3))
    occ.importShapes(str(base_step))
    occ.synchronize()
    after = set(t for _, t in occ.getEntities(3))
    base_tags = sorted(after - before)
    print(f"[merge] base volumes: {len(base_tags)} tags={base_tags}")
    if not base_tags:
        print(f"ERROR: {base_step} imported 0 volumes", file=sys.stderr)
        return 1

    # ── Fuse ──
    rf_dimtags   = [(3, t) for t in rf_tags]
    base_dimtags = [(3, t) for t in base_tags]
    all_dimtags  = rf_dimtags + base_dimtags

    if len(all_dimtags) > 1:
        fused_dt, _ = occ.fuse(
            rf_dimtags, base_dimtags,
            removeObject=True, removeTool=True,
        )
        occ.synchronize()
        print(f"[merge] fused result: {len(fused_dt)} volumes")
    else:
        fused_dt = all_dimtags
        print("[merge] only one input, no fuse needed")

    # ── Verify model has solids ──
    current_vols = occ.getEntities(3)
    print(f"[merge] OCC model now has {len(current_vols)} volumes")
    if not current_vols:
        print("ERROR: no 3D volumes in model after fuse", file=sys.stderr)
        return 1

    # ── heal + deduplicate for clean STEP export ──
    try:
        occ.healShapes(dimTags=current_vols)
        occ.synchronize()
    except Exception as e:
        print(f"[merge] healShapes skipped: {e}")

    occ.removeAllDuplicates()
    occ.synchronize()

    # Final verification
    current_vols = occ.getEntities(3)
    print(f"[merge] after cleanup: {len(current_vols)} volumes")
    for (d, t) in current_vols:
        b = occ.getBoundingBox(d, t)
        print(f"    vol {t}: x=[{b[0]:.4f},{b[3]:.4f}]  "
              f"y=[{b[1]:.4f},{b[4]:.4f}]  z=[{b[2]:.4f},{b[5]:.4f}]")

    # Write result
    out_step.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(out_step))
    print(f"[merge] wrote: {out_step}")

    gmsh.finalize()

    # Sanity: re-open and confirm 3D volumes survived the write
    gmsh.initialize([sys.argv[0]])
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(str(out_step))
    gmsh.model.occ.synchronize()
    check_vols = gmsh.model.occ.getEntities(3)
    print(f"[merge] re-import check: {len(check_vols)} volumes in output STEP")
    gmsh.finalize()

    if not check_vols:
        print("ERROR: written STEP has no 3D volumes — fuse succeeded but "
              "export dropped solids.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
