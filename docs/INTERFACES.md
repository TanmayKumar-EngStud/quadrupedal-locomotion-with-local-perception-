# Interface examples for team review

**Draft, October 5, 2026.** These examples propose the shared interfaces for review by Tanmay, Amedeo and Damien. They are not an implemented API or an approved contract. Current code still loads the original scene schema; do not feed this example bundle into its scene loader.

The complete, valid JSON examples are in [examples/interfaces.json](examples/interfaces.json). The bundle contains separate messages, not one runtime request. They describe one illustrative decision: the robot starts at (0, 0), sees a straight sequence of flat 0.50 m piles separated by 0.03 m gaps, and selects a subgoal 1.59 m ahead toward a destination 8 m away. They are not a complete terrain or benchmark episode.

The Day-1 [abstract simulator](ABSTRACT_SIMULATOR.md) now has internal Python types and a separate replay-log schema. It adapts existing v1 scenes; it does not consume this example bundle or freeze the proposed team contract. Its temporary defaults allow closer-than-1 m recovery targets and use a point-body straight-line support check. These choices are explicit prototype assumptions awaiting controller calibration and team agreement.

## 1. Coordinate and unit convention

- Use SI units: metres, seconds, radians. World/odometry is right-handed, with +z upward. At zero heading, forward is +x and left is +y.
- `world` is the simulator's fixed frame. `odom` is the robot's estimated fixed frame. For the initial perfect-odometry simulation they coincide; they must not silently be treated as identical when odometry errors are later introduced.
- `local_yaw` has its origin at the robot base and rotates with yaw only. It does not roll or pitch with the body. Thus +x is forward, +y left and +z up.
- A pile's `top_center_world_m` is the centre of its top surface, not the centre of its solid collision geometry. `top_size_xy_m` gives side lengths in that top surface's own axes.
- `top_orientation_rpy_rad` defines the top orientation with `R = Rz(yaw) Ry(pitch) Rx(roll)`. Zero orientation means a horizontal top. Positive yaw is counterclockwise viewed from above. Core examples use square tops with yaw zero. Supporting arbitrary yaw would require preserving the top's tangent axes in planner observations as well as its normal.
- Local top normals are unit vectors transformed into the yaw-only frame. For slanted piles, normal = `Rz(-robot_yaw) R_pile [0,0,1]`. A flat top has normal `[0,0,1]`.
- Transform planar targets as `target_local = Rz(-robot_yaw) (target_odom - base_odom)`. Heading error is wrapped to [-pi, pi], then encoded in the order `[sin(error), cos(error)]`.

## 2. Pile description: terrain generator to scene consumers

Each pile carries an opaque ID, top centre, dimensions and orientation. IDs must be stable within an episode. Do not encode hidden terrain parameters in IDs or feed their string values into a learned model.

The full scene may additionally include generator parameters, ground-truth friction, all pile records and episode metadata. Keep these behind simulator/oracle APIs: local planners must not receive hidden geometry, actual friction or ground-truth connectivity. Material/contact settings are simulator configuration, not assumed perception.

Map association uses simulator-stable pile IDs initially, representing perfect data association. State this assumption alongside the perfect local detector; implementing real data association is outside this draft.

## 3. Planner input

`planner_observation` includes robot odometry, the final goal vector, the observed region, visible pile records and a candidate action mask. In this example there are four visible piles. Piles 1.06 and 1.59 m ahead are eligible; the current pile and the pile only 0.53 m away remain perceptual context but are masked from selection under the nominal 1–2 m subgoal rule.

- Visibility uses the horizontal distance from the base to the pile's top centre, at most 2 m. The initial detector has no occlusion. This is a stated simulation assumption, not a realistic sensor guarantee.
- The observed region lets a memory-based planner distinguish observed missing support from unseen space. Boundary cells/footprints must remain unknown unless completely covered by the observation model; do not mark partially observable piles absent merely because their centres are outside the disc.
- `action_mask[i]` refers to `visible_piles[i]`. It must be computed using only current local observations and documented controller limits, never using a full-map oracle route.
- Distance alone is insufficient: a candidate requires an acceptable local approach across intermediate support. A winding graph route is not sufficient evidence that a controller receiving only the endpoint can follow it. The corridor-validity rule is a team decision before implementation.
- The example mask assumes the straight, flat 3 cm-gap corridor is accepted by a provisional geometric rule. It is **not** a measured Go1 success claim.
- Permitted recovery targets closer than 1 m, turning restrictions, and the handling of all-masked candidates must be agreed. Until then, a planner can report `no_candidate` and the harness must log this explicitly rather than invent a target or silently stay in place.
- Raw IDs, episode IDs, decision indices and absolute coordinates used for map bookkeeping are not learned per-pile features. Candidate ordering must be deterministic and identical for scores, labels and masks. Never truncate visible piles silently; if tensors are padded, use a separate padding mask.

Memory is variant-specific. Greedy and L0 receive no visit-history features. L1 retains a GRU state. L2 receives visit counts; L3 uses both. `optional_visit_memory` illustrates a separate variant input. The optimistic planner maintains its own map of observed support, gaps and unknown areas. Reset every memory at episode boundaries.

Proposed neural feature order per pile: relative x/y/z, size x/y, normal x/y/z (8 floats), plus visit count only for the relevant variants. The goal vector is a separate two-float input. Padding capacity, feature scaling and recurrent-state dimensions remain to be agreed with Damien.

## 4. Planner selection and resolved subgoal

The planner selects a candidate ID. Code validates it against the current observation and mask, then resolves its observed coordinates into the odometry frame. This avoids a network inventing arbitrary target coordinates.

`resolved_subgoal` preserves the proposed `{x, y, heading}` interface and adds identifiers, frame, version and `is_final_goal`. For the initial default, heading points from the current base toward the selected pile. An orientation at arrival that anticipates the next turn is not part of this default.

IDs in the examples pair the selection with `episode_id` and `decision_index` to prevent accidental reuse of a stale target. When there is no eligible candidate, return `status: "no_candidate"` with `target_pile_id: null`; do not emit a resolved movement command. The team must decide whether to invoke a recovery action or terminate with a distinct failure reason.

## 5. Executor to controller

The executor keeps the target fixed in odometry coordinates and recomputes the local command every policy step (50 Hz). It owns the 4 s subgoal deadline, decrements `time_left_s`, and asks for another subgoal on arrival within 0.25 m or on timeout. Physics integration and joint servos may run at a higher frequency than the 50 Hz policy.

`is_final_goal` is a proposed addition: it allows the terminal stand-still behaviour to differ from immediate chaining at intermediate targets. Amedeo should confirm whether this is an explicit policy input or handled through his command/reward implementation. Final success needs an agreed support/stability criterion, not just base proximity over a gap.

Proprioception, previous actions and the 1.6 × 1.2 m height scan are produced by the controller environment, not sent by the planner. Freeze their tensor ordering, scan origin, extents, flattening order and treatment of no-hit rays separately with Amedeo before training. No trained policy exists in the current repository.

## 6. Review checklist

- **All:** confirm SI units; yaw-only local frame; perfect odometry/detection assumptions; top-centre/size/orientation definitions.
- **Amedeo:** confirm pile geometry export, locally feasible approach rule, controller command, final-goal handling and future hop-calibration inputs.
- **Damien:** confirm pile feature order, normal-based slant representation, variable-size candidate sets/masks, and ID-to-label mapping.
- **Tanmay:** freeze the agreed schema, candidate validation, coordinate conversions, benchmark manifests, observation boundaries and logging.

Keep the draft version until the team agrees. Do not start treating these examples as a silently approved replacement for teammate interfaces.
