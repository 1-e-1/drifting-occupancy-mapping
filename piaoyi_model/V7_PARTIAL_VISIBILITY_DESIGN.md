# V7 partial-visibility reconstruction

## Observation contract

- The generator receives only a 16-frame RGB-D history, calibrated camera poses, ray/free-space evidence and its recurrent occupancy memory.
- It does not receive the number of objects, object IDs, centers, sizes or boxes.
- Simulator object metadata is retained only to construct continuous occupancy/SDF training targets and offline metrics.

## Collection policy

- Object count, size, spacing, position and 6-D pose are sampled per episode.
- Object count is frozen before camera/layout retries so acceptance cannot bias the dataset toward easier low-count scenes.
- The initial frame must contain aggregate object evidence and at least one visible object, but need not show every object.
- Later views may contain partial or very weak object evidence.
- No single frame must show every object, but the complete 120-frame history must contain evidence for every physical object; this is a dataset learnability check, not an inference input.
- A trajectory is rejected only when object-free frames exceed 20% or an object-free run exceeds 15 consecutive frames, or when camera-clearance, collision and motion-diversity checks fail.

## Geometry inference and planning safety

1. The drifting generator predicts an occupied point/volume distribution without object slots.
2. Density filtering, free-space carving and morphology produce a non-convex occupancy shell.
3. Connected components determine the inferred object count.
4. Components separated by less than 0.10 m may share one conservative oriented box. No true object count is consulted.
5. For trajectory planning, the non-convex occupancy is inflated by the robot-link/tool clearance radius. Corridors narrower than twice that radius disappear even if visual clustering kept separate components.

This intentionally permits two nearby physical objects to be represented as one obstacle while retaining complex outer boundaries for safe visual-servo and motion-planning consumers.
