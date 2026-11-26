# V1 3D campaign ledger (trapezoid blade, 33-knob codec)

Backup of the three Colab shard ledgers of the Stage-3 3D CFD campaign (Aug 2026), recovered from
Drive (`My Drive/fanopt`) into `print_ready/drive_recovery/campaign_trapezoid/` on 2026-11-14 and copied
here on 2026-11-25 so the V2 training data is version-controlled.

- 838 rows, 805 designs; 751 with a finite `j_fan` (87 NaN = failed runs); sources: 179 `sobol`,
  610 `bo`, 49 `explore`.
- Row fields: `design_hash`, `vector` (33 codec values, `fanopt.bo.blade_codec.decode`), `session_id`,
  `source`, `timestamp_iso`, `j_fan` (signed cycle-mean CFz, scaled coefficient — not newtons, see
  `docs/V2_plan.md` §1 gap 4), `mass_kg` (solid blade at the searched thickness, PETG), `deflection_m`
  (analytic plate proxy), `blade_count`, `dispatch_s`, `finish_s`, `inflight_at_dispatch`.
- Read with `fanopt.cfd.blade_verify.load_campaign_rows(<this dir>)`.

`data/stage4_to/` holds the Stage-4 TO summaries (8 designs, Aug 2026) and design 00's per-design
report; the density/centroid arrays are on Drive (`My Drive/fanopt/stage4_blade_to`) and in
`print_ready/drive_recovery/`.
