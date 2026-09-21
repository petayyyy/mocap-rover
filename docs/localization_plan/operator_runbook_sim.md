# Simulation operator runbook

## Preconditions

Use the repository root and Python 3. Do not attach physical cameras or enable
trigger/udev/libcamera. `hardware_verified` must remain false.

## Contract and regression checks

```text
python3 -m unittest discover -s tests -v
gz sdf -k worlds/mocap_arena.sdf
python3 - <<'PY'
from simulation.pipeline import run, evaluate_samples
r = run(2, return_samples=True)
print(r)
print(evaluate_samples(r))
PY
```

## Gazebo baseline smoke check

```text
./scripts/run.sh -s --headless-rendering
/usr/bin/python3 scripts/check_sim.py --measure-seconds 3
```

Expected evidence is six RGB streams and both world poses. Wall FPS is reported
separately from simulation FPS; the known shutdown GIL abort and NVIDIA library
workaround are limitations, not hidden success.

## Dashboard

After `check_sim.py` writes snapshots, start the localhost-only dashboard with:

```text
python3 scripts/serve_dashboard.py --snapshot-dir /tmp/mocap-s15-preview-20260921 --port 8080
```

It exposes `/api/status`, `/api/cameras`, `/api/previews` and six `/preview/`
URLs. Browser disconnect must not be treated as a tracker stop. This is a
simulation snapshot walkthrough, not a SIM_ACCEPTED performance claim.

## Acceptance discipline

Read `sim_acceptance_report.md` and `STATUS.md`. Never set `SIM_ACCEPTED` based
on synthetic-only rates or ground-truth evaluator output. H01–H04 require a
separate explicit user request after simulation acceptance.
