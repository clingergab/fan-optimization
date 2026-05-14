# Spike 0.6 — Colab Pro compute-budget probe + M3 local-pipeline sub-spikes

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.6`.

**Why this exists.** Phase 4 and Phase 5 schedule a large number of Tier-1
(3D unsteady, ~500K cells, 5 pitching cycles at `dt = T/200`) SU2
evaluations. The per-evaluation wall-time and compute-unit consumption on
Colab Pro CPU vs. G4-class GPU is the dominant uncertainty in the Phase 4
budget, and the MacBook M3's usability for *any* local SU2 or FEA work
falls out of the same set of measurements. Spike 0.6 records both, but
the parent spike is **calibration, not a gate** — only the two sub-spikes
(0.6a, 0.6b) gate downstream phases.

| Strand | What it gates | Pass criteria |
|---|---|---|
| Calibration probe | nothing — informational | n/a |
| Sub-spike 0.6a | local-M3 SU2 use (Phase 0 `smoke_test.py`) | wall-time <= 15 min AND `J_fan_steady_proxy` finite |
| Sub-spike 0.6b | local-M3 FEA (Phase 5 step 59.5 / step 64.5) | wall-time <= 2 min AND tip deflection within 5% of analytic |

---

## Apparatus

You need:

- **Colab Pro account** with at least one G4-class GPU runtime available
  (per the Round-9 hardware lock — T4 is insufficient for Phase 5
  verification at PyFR p=3 anyway, so we standardise on G4 here).
- **MacBook M3** with the project Python environment installed.
  Optional local installs:
  - `SU2_CFD` on `PATH` for sub-spike 0.6a.
  - `dolfinx` (FEniCSx) importable in the project Python environment for
    sub-spike 0.6b. (CalculiX is an equivalent fallback; the protocol below
    assumes FEniCSx.)
- A stopwatch or the Colab cell's built-in `%%time` magic for the Colab
  rows.

---

## Step 1 — Colab CPU 3D unsteady probe

Workload: **500K cells, 5 pitching cycles, dt = T/200** — the Tier-1
locked config in §9.4.1.

1. Spin up a Colab Pro **CPU** runtime (no accelerator).
2. Mount the project, generate (or download) the 500K-cell baseline mesh.
3. Run the unsteady SU2 cfg (`configs/su2/tier1_unsteady.cfg.j2` rendered
   with the baseline geometry). Time it end-to-end with the cell's
   `%%time` magic, including I/O.
4. Record the **wall-time in seconds** and the **Colab compute units
   consumed** (visible in Colab's "Resources" / billing panel; subtract
   the pre-run total from the post-run total).
5. Add one row to `data/spike_0_6/budget.csv`:

```
platform,workload,wall_time_s,cu_consumed,cells,notes
colab_pro_cpu,3d_unsteady_500k_5cycles_dtT200,<wall>,<cu>,500000,<notes>
```

Note: the unsteady cfg uses `MACH = 1e-9` with the
`FREESTREAM_OPTION = FREESTREAM_VELOCITY` override (Round-9 HIGH-12 lock).
Do NOT edit the cfg — the budget number must reflect the production
config.

## Step 2 — Colab GPU G4-class 3D unsteady probe

Same workload, same cfg, same mesh — only the runtime changes.

1. Switch the Colab runtime to **G4 GPU** (95 GB).
2. Re-run the unsteady cfg from Step 1.
3. Record wall-time and CU consumed.
4. Add the second row to `data/spike_0_6/budget.csv`:

```
colab_pro_g4_gpu,3d_unsteady_500k_5cycles_dtT200,<wall>,<cu>,500000,<notes>
```

Two budget rows is the minimum the aggregator expects; add more rows if
you want to record A100 / multi-cycle / smaller-mesh variants for the
Phase 4 cost model.

## Step 3 — Sub-spike 0.6a (M3 SU2 Tier-1 end-to-end)

On the M3:

```
python scripts/run_spike_0_6a.py
```

The runner checks `which SU2_CFD` first. If SU2 is not installed locally,
it prints a clear "SU2 not installed locally — see
docs/spike_0_6_protocol.md §06a fallback" notice and exits 0 with a
"gated-on-availability" marker — the aggregator then sees no 0.6a row and
reports the sub-spike as skipped.

If SU2 is installed, the runner times the Tier-1 pipeline (CadQuery ->
Gmsh 2D corrugated slice -> SU2 2D steady -> `j_fan.py`) end-to-end and
writes `data/spike_0_6/06a.csv` with:

- the total wall-time in seconds
- `J_fan_steady_proxy` from `j_fan.py`
- per-stage breakdown rows (cadquery / gmsh_2d / su2_2d_steady / j_fan)

**Pass criterion (gate for any local-M3 SU2 use):**

1. wall-time <= 15 min
2. `J_fan_steady_proxy` is finite (NaN / inf -> fail)

**§06a fallback** — if 0.6a fails (or SU2 is unavailable / takes too
long): shift the Phase 0 smoke pipeline to a Colab Pro CPU session. The
M3 retains its other roles (geometry generation in CadQuery, mesh
quality-control review in Gmsh, Fusion-360 inspections, IMU data logging
for Spike 0.3 / Phase 6) — only the SU2 invocation moves to Colab.

## Step 4 — Sub-spike 0.6b (M3 FEA cantilever)

On the M3:

```
python scripts/run_spike_0_6b.py
```

The runner checks for `dolfinx` (`importlib.util.find_spec("dolfinx")`).
If FEniCSx is not installed, it prints "FEniCSx not installed; sub-spike
0.6b can't run locally — see protocol §06b fallback" and exits 0.

If FEniCSx is installed, the runner solves a 1D Euler-Bernoulli
cantilever with the locked spec (PETG, b=12 mm, h=2 mm, L=200 mm, P=5 N,
E = 1300 MPa) and writes `data/spike_0_6/06b.csv` with:

- wall-time in seconds
- measured tip deflection in metres
- the (P, L, E, I) used so the aggregator can re-derive the analytic
  reference independently

The analytic reference is `delta = P L^3 / (3 E I)` with
`I = b h^3 / 12`. For the locked spec:
`I = 0.012 * 0.002^3 / 12 = 8e-12 m^4`,
`delta_analytic = 5 * 0.200^3 / (3 * 1.3e9 * 8e-12) ≈ 1.282 m`.
(This is a beam-theory limit — the cantilever has gone well past
small-deflection validity. The number is what Euler-Bernoulli gives;
sub-spike 0.6b is checking that the FEA stack agrees with the Euler-
Bernoulli analytic to within 5%, not that the physical PETG beam would
actually deflect 1.28 m — at 5 N a 2 mm-thick PETG rib has yielded long
before then. The cross-check is a numerical sanity gate, not a strength
prediction.)

**Pass criterion (gate for Phase 5 step 59.5 / step 64.5 local FEA):**

1. wall-time <= 2 min
2. `|measured - analytic| / analytic * 100 <= 5%`

**§06b fallback** — if 0.6b fails (or FEniCSx is unavailable / the cross-
check exceeds 5%): move Phase 5 step 64.5's combined-blade structural
FEA to a Colab Pro CPU session. The M3 retains its non-FEA roles
(geometry, mesh-QC, Fusion-360, IMU) as in the §06a fallback.

## Step 5 — Aggregate and write `results.json`

```
python scripts/run_spike_0_6.py \
  --budget-csv data/spike_0_6/budget.csv \
  --06a-csv data/spike_0_6/06a.csv \
  --06b-csv data/spike_0_6/06b.csv \
  --out data/spike_0_6/results.json
```

The aggregator reports:

| Field | Meaning | Pass criterion |
|---|---|---|
| `budget_entries[i].wall_time_s` | Colab wall-time (CPU / GPU) | (informational — calibration) |
| `budget_entries[i].cu_consumed` | Colab compute units consumed | (informational) |
| `sub_06a.passed` | M3 Tier-1 wall-time <= 15 min AND J_fan finite | gate for local-M3 SU2 |
| `sub_06b.passed` | M3 FEA wall-time <= 2 min AND tip-pct <= 5% | gate for local-M3 FEA |
| `overall_passed` | Always True (calibration framing) | n/a |

Exit codes:

- `0` — both supplied sub-spikes passed (or absent, treated as skipped).
- `1` — at least one supplied sub-spike failed its gate.
- `2` — input error (missing file, missing column, non-numeric).

---

## Diagnostics if either sub-spike fails

| Symptom | Likely cause | Fix |
|---|---|---|
| 0.6a wall-time > 15 min | M3 thermal throttling | Run with the laptop on a cooling pad; close other apps |
| 0.6a wall-time > 15 min | Mesh > 500K cells slipped in | Verify the 2D slice has the expected cell count |
| 0.6a `J_fan_steady_proxy` NaN | SU2 didn't converge | Inspect SU2 history.csv; bump CFL or iters |
| 0.6a SU2 missing | M3 has no SU2 binary | Use the §06a fallback (Colab CPU smoke) |
| 0.6b wall-time > 2 min | Too-fine mesh | Coarsen the 1D mesh to <= 100 elements |
| 0.6b tip-pct > 5% | Wrong cross-section in solver | Verify `I = b h^3 / 12` (bending about b-axis) |
| 0.6b FEniCSx missing | conda env not set up | Use the §06b fallback (Colab CPU FEA) |

---

## What this rig is reused for

- **Phase 4 budget tracker** consumes `budget_entries` to project total
  Colab CU consumption for the BO outer loop.
- **Phase 5 step 59.5 / step 64.5** consults `sub_06b.passed` to decide
  whether the structural FEA runs locally on the M3 or on Colab.
- **Phase 0 `smoke_test.py`** consults `sub_06a.passed` for the same
  reason on the SU2 side.
