# V7 Visual-servo guidance contract

This is the retained adapter for the first-order V7 spatiotemporal model.
It does not issue hardware commands and is not a collision-safety certificate.

## Inputs and outputs

- RGB-D history: `[B,16,5,128,128]` after preprocessing.
- Camera poses: `[B,16,7]`; calibrated intrinsics: `[B,3,3]`.
- Generated particles: `[B,2048,3]`; observed points: `[B,8192,3]` plus mask.
- Tri-state memory: `[B,1728,3]` (occupied/free/unknown).
- `extract_environment_outline`: non-convex shell, frontier, components and optional conservative boxes.
- `build_visual_servo_guidance`: `[B,22]` controller features, world-frame target and validity.

The feature vector concatenates 9 reconstruction statistics, 9 outline statistics,
3 target-direction coordinates and 1 distance. The generator never receives the
true object count or the simulator boxes. Ground truth is for supervision and evaluation only.

## Execution

1. Fuse the synchronized RGB-D history with `reconstruct_occupancy_v3`.
2. Extract the measured/predicted outline with `outline.py`; the V7 evaluation
   uses adaptive particle splatting on a 48^3 grid and threshold 0.85.
3. Build guidance and check validity, finite values, observation age and robot-specific limits.
4. A downstream controller proposes camera motion; a separate full-arm checker
   must validate joint limits, environment collisions and inter-arm collisions.

The control interface is designed around 15 Hz, but the retained V7 selection record
reports approximately 104.6 ms end-to-end guidance on its historical test setup.
It therefore does NOT establish that every control tick can synchronously rebuild the outline.
Use asynchronous reconstruction and explicit age gates, with fresh validation after hardware changes.

## Retained evaluation

See `checkpoints/drifting_v7/selection.json` from the unified root for the 96-window
historical V7 evaluation. Boundary coverage is about 63.01%, shell precision 31.60%,
boundary Chamfer 0.1768 m. These are occupied-volume outline metrics, NOT the much
smaller measured-surface errors reported for the non-neural Planning v3 route.

The PPO and World Model in `checkpoints/drifting_controller/` originate from the
previous V6 controller training. V7 closed-loop deployment has not been validated.
The retained source supports retraining; preservation of those weights is not proof
that their dynamics/belief predictions remain correct with the V7 generator.

## Safety

Generated hidden geometry is a hypothesis, not measured free space. Hold on stale,
invalid or insufficient-confidence guidance. Require independent emergency stop,
watchdog, hand-eye calibration, trajectory checks and sensor/latency validation.

See [V7_PARTIAL_VISIBILITY_DESIGN.md](V7_PARTIAL_VISIBILITY_DESIGN.md) for data
visibility constraints. Temporary GIF/PNG demos have been deleted; retained source
and numeric reports are the authoritative artifacts.
