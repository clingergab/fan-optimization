# fan-optimization

A 3D-printed folding hand fan designed by optimization: Bayesian optimization over 3D unsteady CFD picks
the blade shape, then 3D topology optimization hollows each winning blade into a printable part.

<p align="center">
  <img src="docs/images/fan_deployed.png" alt="Deployed 12-blade fan" height="400">
  &nbsp;&nbsp;
  <img src="docs/images/fan_folded.png" alt="Folded 12-blade stack" height="200">
</p>

## Pipeline

1. **Geometry** — CadQuery blade from a 33-parameter codec (rib wave, panel shape, thickness); 12 blades must fold into a ≤ 90 mm stack.
2. **CFD** — SU2 unsteady 3D simulation of the ±40°, 2 Hz stroke; objective `J_fan` = cycle-mean thrust.
3. **Optimization** — BoTorch (qLogNEHVI + TuRBO) over wind ↑, mass ↓, deflection ↓, run across parallel Colab sessions.
4. **Verification** — top designs re-run at fine CFD resolution.
5. **Topology optimization** — per-blade 3D SIMP carves the interior while keeping the aero surface.
6. **Print** — marching cubes → watertight STL → Bambu Studio.

## Status

V1 is done computationally: three blades are ready to print ([print guide](docs/print_guide.md),
[STL files](https://github.com/clingergab/fan-optimization/releases/tag/v1.0-print),
[3D previews](docs/models/)). Next is printing and a blinded feel test against a flat-blade baseline.
V2 plans are in the [backlog](docs/V2_backlog.md).

<p align="center"><img src="docs/images/v1_blades.png" alt="The three V1 blades" width="600"></p>

## Quick start

```bash
conda env create -f environment.yml && conda activate fanopt
pip install -e ".[dev]"
pytest
```

CFD and topology-optimization runs live in [`notebooks/`](notebooks/README.md) (Colab). Design decisions
are in [`docs/adr/`](docs/adr/README.md).

## License

MIT
