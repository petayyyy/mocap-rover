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

Start the localhost-only handler from Python with `serve("127.0.0.1", 8080)`;
`/api/status` is diagnostics/settings metadata only. Browser disconnect must not
be treated as a tracker stop. The current dashboard is a smoke-test endpoint,
not a SIM_ACCEPTED visual walkthrough.

## Acceptance discipline

Read `sim_acceptance_report.md` and `STATUS.md`. Never set `SIM_ACCEPTED` based
on synthetic-only rates or ground-truth evaluator output. H01–H04 require a
separate explicit user request after simulation acceptance.
