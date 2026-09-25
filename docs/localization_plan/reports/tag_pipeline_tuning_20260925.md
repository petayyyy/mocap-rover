# AprilTag pipeline tuning from coverage run 20260925_001059

This tuning pass is based on the recorded CSV/JSONL artifacts only. Gazebo was
not started.

## Baseline evidence

- 50,742 camera frames and 4,484 recorded observations.
- Raw tag detection in at least one camera: 91.27% of exposures.
- Valid PnP in at least one camera: 49.41% of exposures.
- Fused estimate accepted: 40.71% of exposures.
- Fused XY error: p50 0.085 m, p95 0.368 m.
- Fused yaw absolute error p95: 15.52 degrees.
- Median Z bias per camera was approximately +0.28 to +0.35 m.
- The two oblique center cameras had opposite systematic X biases of
  approximately -0.164 m and +0.158 m.

The Z and opposite X biases are consistent with selecting the wrong branch of
the two-solution planar IPPE pose, not with a constant camera translation
error. The old runtime also retained whichever synchronous camera worker
reached the fusion lock first.

## Applied runtime defaults

- supported families: AprilTag 36h11 and ArUco DICT_4X4_50;
- detector profile: `coverage`, with allowed-ID filtering;
- detector scale: 1.0 for both the current stream and native 640x480;
- minimum PnP quality: 0.07;
- minimum detected side: 8 px;
- maximum reprojection RMS: 5.0 px;
- maximum reconstructed rover tilt: 40 degrees;
- synchronous-camera selection: quality, then reprojection, then tag size;
- empirical XY sigma: `max(0.025, 0.018 * view_factor / quality)` metres,
  where `view_factor=1.6` for cameras 5 and 6;
- yaw sigma: quality-dependent, limited to 2--20 degrees;
- yaw innovation gate: 60 degrees.

## Algorithm changes

1. Evaluate both `SOLVEPNP_IPPE_SQUARE` solutions.
2. Refine both with LM and select the solution whose reconstructed rover Z
   axis is closest to the arena Z axis.
3. Reject degenerate quadrilaterals, excessive reprojection error and
   non-planar solutions before fusion.
4. Replace first-thread-wins fusion with a bounded synchronous timestamp
   arbiter. A stalled camera cannot block more than four pending timestamps.
5. Record PnP rejection reasons, candidate count, reconstructed tilt and the
   covariance used by fusion.
6. Use raw detection first, then bounded CLAHE/Otsu fallbacks only for a
   plausible rejected quad; this protects temporal coverage on empty cameras.

## Offline verification

- Full unit suite: 97 tests passed, 2 skipped because optional Gazebo snapshots
  were unavailable.
- Synthetic projection with the recorded six camera calibrations, a planar
  0.40 m tag and 0.2 px corner noise:
  - old single-branch IPPE: XY p50 0.0148 m, p95 4.44 m;
  - new planar branch selection: XY p50 0.0069 m, p95 0.0439 m;
  - new Z absolute error: p50 0.0063 m, p95 0.0228 m.

The synthetic comparison validates branch selection but is not a replacement
for a new coverage run. The next Gazebo run must compare detection rate, PnP
rate, accepted rate, XY/Z/yaw errors, rejection reasons and processing latency.
