# PPT Slide-by-Slide Defense Document
## Localisation-Uncertainty-Aware Deep Reinforcement Learning for Active SLAM on UAVs in GPS-Denied Environments

**Authors:** Anjali Gupta (23095013), Rehaan Goenka (23095082)
**Supervisor:** Dr. Om Jee Pandey
**Department of Electronics Engineering, IIT (BHU) Varanasi**

---

This document goes through every slide in the presentation, every term that appears on the slide, and explains it in depth so you can answer any cross-question. **Read this slide-by-slide, in order.** Each slide section ends with a list of *Likely Questions* a professor might ask, with model answers.

---

# Slide 1 — Title Slide

## What's on the Slide

- "EC-392 : UG PROJECT-I"
- "LOCALISATION-UNCERTAINTY-AWARE DEEP REINFORCEMENT LEARNING FOR ACTIVE SLAM ON UAVS IN GPS-DENIED ENVIRONMENTS"
- Group Members: Anjali Gupta (23095013), Rehaan Goenka (23095082)
- Supervisor: Dr. Om Jee Pandey
- IIT (BHU) Varanasi

## Term-by-Term Breakdown

### "EC-392 : UG PROJECT-I"
EC-392 is the course code for the first Undergraduate Project in the Electronics Engineering curriculum at IIT BHU. It's a 6-credit project where students conduct independent research under a faculty supervisor.

### "Localisation-Uncertainty-Aware"
This means our method **explicitly accounts for how confident the robot is in its own position estimate** while making decisions. "Localization uncertainty" is the spread of the probability distribution over possible robot positions — we encode this as a scalar σ_t (sigma) and the policy learns to keep σ_t small.

### "Deep Reinforcement Learning" (DRL)
- **Reinforcement Learning (RL)**: a machine-learning paradigm where an agent learns a behavior (policy) by trial-and-error interactions with an environment, getting numerical rewards.
- **Deep**: the policy is represented by a deep neural network (multiple hidden layers) — specifically a CoordConv-CNN + LSTM + MLP in our work.

### "Active SLAM"
- **SLAM**: Simultaneous Localization and Mapping — building a map of an unknown environment while simultaneously estimating the robot's location in that map.
- **Active**: the robot actively *chooses where to go* to make SLAM work better, rather than passively executing a pre-planned path. Active SLAM jointly optimizes (i) where to move, (ii) what to map, (iii) how to reduce position uncertainty.

### "UAVs" (Unmanned Aerial Vehicles)
Aircraft without an onboard human pilot — controlled remotely or autonomously. We specifically use a **quadrotor** UAV (4-rotor configuration) because it can hover and change direction quickly, ideal for indoor exploration.

### "GPS-Denied Environments"
Places where Global Positioning System satellite signals are blocked or unreliable:
- Inside warehouses (metal shelving blocks signals)
- Tunnels, mines, basements (no line-of-sight to satellites)
- Disaster zones with collapsed structures
- Indoor military operations

In GPS-denied environments, the UAV cannot directly know its position from satellites — it must localize itself using onboard sensors (LiDAR, IMU). This is why SLAM is needed.

## Likely Questions

**Q: Why is GPS-denied environments a hard problem?**
A: Without GPS, the UAV has no absolute reference for its position. It must build its own map AND estimate its position within that map — both at once, from noisy sensor data, which is the SLAM problem. Errors compound over time (drift), and without intervention, position estimates can become arbitrarily wrong.

**Q: Why a quadrotor specifically?**
A: Quadrotors can hover and change direction instantly, making them ideal for tight indoor spaces. Fixed-wing UAVs need forward speed; we want flexibility. Helicopters are mechanically complex; quadrotors are simple. Most importantly, quadrotors can fly at *any altitude* within a layer, which is what enables our multi-altitude active SLAM.

**Q: What does "active" mean specifically in Active SLAM?**
A: In passive SLAM, a human or fixed plan moves the robot, and SLAM just observes and computes. In active SLAM, the robot *chooses* its trajectory based on what it expects each move will reveal — it might revisit somewhere on purpose to trigger a loop closure, or push into unknown space to see new things. The robot has agency over its own data collection.

---

# Slide 2 — Motivation

## What's on the Slide

- UAVs are used in warehouses, tunnels, disaster zones for mapping unknown environments
- Task requires simultaneous exploration + accurate localization (Active SLAM)
- Challenge: need to maximize coverage while minimizing localization uncertainty
- Two photos: a damaged vehicle in a disaster scene; a UAV near a cliff

## Term-by-Term Breakdown

### "Warehouses"
Large indoor industrial spaces with shelving, conveyor belts, storage racks. UAVs are used for inventory management, package routing, defect inspection. The challenge: GPS doesn't work indoors due to metal/concrete structures.

### "Tunnels"
Underground passages — mines, sewers, utility tunnels, transit tunnels. UAVs inspect for cracks, leaks, structural damage. Tight spaces with no GPS make this a SLAM showcase.

### "Disaster Zones"
Earthquakes, floods, building collapses, war zones. UAVs scout for survivors, map damage, identify hazards before sending in human responders. Time-critical — UAVs must rapidly cover ground while handling unknown obstacles.

### "Mapping Unknown Environments"
The robot has no prior map. It must construct one from scratch using its sensors. The map could be:
- A 2D occupancy grid (free / occupied / unknown cells)
- A 3D voxel grid
- A topological graph
- A point cloud
We use a **5-channel 48×48 map tensor** at each of 4 altitudes (so effectively a 48×48×4 voxel grid).

### "Exploration"
Moving into previously unobserved space. Increases the *coverage* of the map.

### "Localization"
Knowing where you are. Two flavors:
- **Global localization**: from scratch — figure out where you are in the map.
- **Pose tracking**: assuming you knew where you were, update as you move.
SLAM does both. Localization quality is measured by the *covariance* of the pose estimate.

### "Active SLAM" (formal definition)
Active SLAM is the joint optimization of:
1. The exploration trajectory (where to go)
2. The map estimate (what's where)
3. The pose estimate (where am I)
4. The pose uncertainty (how confident am I in my pose)
A formal POMDP framing makes "minimize uncertainty" an explicit objective alongside coverage.

### "Coverage"
Fraction of the true free space that the robot has observed and confirmed as free or occupied. Mathematically:
Cov(s) = (# observed cells) / (# true free cells in M*)
We report coverage as a percentage. Our headline is **91.90% coverage**.

### "Localization Uncertainty"
The spread of the robot's belief about its own pose. In SLAM, this is captured by the **pose covariance matrix Σ**. The trace of Σ (sum of diagonal entries) is a single scalar summary — bigger trace = more uncertain.
In our work: we use a *scalar surrogate* σ_t ∈ [0, 4] that mimics how Tr(Σ) would behave (grows with motion, shrinks on loop closure).

## Likely Questions

**Q: Why is balancing coverage and localization uncertainty hard?**
A: They pull in opposite directions. Pure coverage says "always go to new territory" — but new territory means you accumulate drift in your position estimate (because the SLAM odometry chain gets longer without revisiting known places). Pure uncertainty minimization says "stay in known space and revisit landmarks" — but you stop exploring. Active SLAM must trade these off, which classical methods do via hand-tuned weights and our method does via a learned policy.

**Q: Why not use GPS + IMU dead reckoning?**
A: Dead reckoning (using IMU velocity integration) accumulates drift fast — a few meters per minute. That's fine for short missions but useless for long indoor exploration. SLAM uses the sensor returns themselves to constrain the pose estimate, which is far more accurate.

**Q: Are there other applications you're targeting?**
A: Yes — search-and-rescue, agricultural drone surveys, military reconnaissance in GPS-jammed environments, autonomous mapping of construction sites. The warehouse setting is our concrete instance, but the methodology applies broadly.

---

# Slide 3 — Research Gap

## What's on the Slide

Three categories with images:
- **Classical Methods** (top-left image: 2D map with drone path):
  - Mostly 2D planning
  - Altitude handled manually
- **DRL Methods** (middle image: 3D mapped scene with drone):
  - Learns exploration well
  - Not altitude-aware
  - Does not handle uncertainty explicitly
- **Real-World Need** (bottom image: multi-layered 3D warehouse):
  - (implicit visual showing layered structure)
- **Core Gap**:
  - No method jointly optimizes coverage, localization uncertainty, and altitude decisions
  - Poor altitude handling leads to incomplete coverage + higher uncertainty
- **Problem Statement** (boxed):
  - Design a single policy that intelligently decides movement + altitude to maximize coverage and minimize localization uncertainty.

## Term-by-Term Breakdown

### "Classical Methods"
Hand-engineered active-SLAM planners. Examples:
- **Frontier-based exploration** (Yamauchi 1997): pick the closest unknown frontier, go there.
- **Nearest Frontier**: variant that scores frontiers by distance.
- **Information-theoretic methods**: pick the action that maximizes expected information gain (e.g., Shannon mutual information).
- **Sampling-based**: RRT (Rapidly-exploring Random Trees), PRM.
- **Potential field**: virtual repulsive forces from obstacles, attractive forces toward frontiers.

These methods use hand-crafted cost functions, often well-understood and easy to debug, but **brittle** — the trade-off weights need tuning per environment.

### "2D Planning"
Most classical methods plan in a single horizontal plane (the flight altitude). Altitude is treated as a separate, simpler problem (often: "stay at flight altitude unless commanded otherwise"). This works for ground robots but wastes UAV capability — we have a third dimension.

### "Altitude handled manually"
Even when classical methods support altitude changes, they require a human-tuned schedule: "fly low for the first 5 minutes, then high, then mid". The altitude policy is not learned and not adapted to the environment.

### "DRL" (Deep Reinforcement Learning) Methods
Methods that learn exploration policies with deep neural networks. Examples:
- **Chen et al. 2024**: LiDAR-based DRL for active SLAM
- **Zhao & Hwang 2024**: DDPG for indoor active SLAM
- **MA-SLAM**: structured map representation for ground-robot DRL
- **NavRL**: PPO for safe UAV flight (but not active SLAM)
- **Curiosity-driven exploration** (Botteghi 2021)

### "Learns exploration well"
DRL methods often achieve high coverage in their target environments — comparable to or better than classical baselines.

### "Not altitude-aware"
Existing DRL methods either:
- Use 3D continuous actions but in environments where altitude doesn't really matter (drone racing, obstacle avoidance), OR
- Plan on a single 2D slice of a 3D environment

None of them treat altitude as an *explicit, discrete decision variable* fed back into the observation, which is what we do.

### "Does not handle uncertainty explicitly"
Most DRL exploration methods reward coverage (or information gain) but **don't have a separate signal for localization uncertainty**. This means the policy might explore a lot but accumulate huge drift — the resulting map is geometrically inconsistent.

### "Core Gap"
Our positioning: **no prior method jointly addresses (i) discrete altitude decisions with dwell-gated layer switches, (ii) dense covariance-trace reward shaping that turns sparse loop closures into a trainable signal, and (iii) recurrent early fusion of map and scalar state in a single learned policy**.

### "Problem Statement"
A precise, single-sentence formulation of what we're solving:

> Design a single policy π_θ(a_t | o_t, h_t) that, given partial observations o_t = (M_t, v_t) and recurrent memory h_t, outputs an action a_t ∈ [-1,1]³ (continuous horizontal movement + discrete altitude vote) so that the expected episode-end coverage Cov(s_T) is maximized while the expected uncertainty σ_T is minimized.

## Likely Questions

**Q: Aren't there altitude-aware classical methods?**
A: Some 3D frontier methods exist (e.g., octomap-based explorers), but they treat altitude as continuous z and use hand-tuned heuristics for "go up vs. go down". None decompose altitude into discrete layers with a learned dwell-gated discrete decision.

**Q: How is your novelty different from MA-SLAM (Yin et al.)?**
A: MA-SLAM contributes a structured *map representation* for large-scale DRL exploration, but it operates on ground robots in a single plane. We're orthogonal: same general DRL framing, but our novelty is the multi-altitude POMDP decomposition + the Δσ uncertainty-shaping reward. We could combine the two.

**Q: What does "joint optimization" mean concretely?**
A: A single neural network outputs all three decisions at once: dx, dy (horizontal), dz (altitude vote), conditioned on the same observation and trained against a *single composite reward* that includes coverage, breadth, uncertainty, and shaping terms. There's no module hierarchy ("first decide altitude, then move horizontally") — the policy reasons over all three jointly.

---

# Slide 4 — System Model

## What's on the Slide

A three-panel figure (the system_model.png we corrected) plus three bullet points:
- An onboard 2D LiDAR scanner feeds the SLAM back-end
- Outputs are summarized into observation o_t consumed by the RL policy
- Emits action a_t closing the perception–action loop

The figure itself shows three panels:
- **(A) Multi-Altitude Warehouse**: 4 layered grids at 1m, 2m, 3m, 4m, with a quadrotor at 3m, LiDAR rays, dz altitude vote arrow
- **(B) SLAM Back-End**: 3D occupancy map M_t, pose estimate (x̂_t, ŷ_t, ẑ_t), uncertainty surrogate σ_t ∈ [0,4]
- **(C) RL Policy and Action Loop**: 5-channel map tensor + 10-D scalar input → CoordConv-LSTM network → Action (dx, dy, dz)

## Term-by-Term Breakdown

### "Closed loop system"
A system where the *output* of one component feeds back as *input* to another, forming a cycle. Here: action → environment → sensor → SLAM → observation → policy → action. The robot's actions affect its future observations, which affect its future actions.

### "Onboard 2D LiDAR scanner"
- **Onboard**: sensor mounted on the UAV itself (not external, not off-board).
- **2D LiDAR**: scans in a horizontal plane only (one layer of data per scan), unlike 3D LiDAR which has multiple stacked beams.
- **Scanner**: rotates 360° and fires laser pulses to measure distance.
We use a 2D LiDAR rather than 3D because: it's lighter, cheaper, lower power, and our environment is 2.5D (separate altitude layers each treated as 2D).

### "Multi-altitude warehouse" (Panel A)
Our environment partitions a warehouse into 4 distinct horizontal layers at heights {1, 2, 3, 4} meters. Each layer has its own 48×48 grid of cells. Some shelves and obstacles only exist at specific altitudes (you can fly *over* low shelves at 4m). The total environment is 48×48×4 = 9,216 voxels.

### "dz altitude vote" (red arrow in Panel A)
The third component of the action — the policy's vote on whether to change altitude. Not a continuous velocity command on z, but a discrete "should I go up, down, or stay?" decision encoded as a value in [-1, +1]:
- dz > 0.3 → go up one layer
- dz < -0.3 → go down one layer
- otherwise → stay at current layer

### "360° LiDAR (36 beams)"
The LiDAR sweeps a full circle (360°) at the current altitude. We discretize this into 36 beams (one every 10°). Each beam fires up to 14 cells away, returning the distance to the nearest obstacle. We model this in code as:

```
for ray_idx in range(36):
    angle = 2π * ray_idx / 36
    walk along (cos angle, sin angle) until obstacle or max range
```

### "SLAM Back-End" (Panel B)
The component that *processes* the LiDAR returns and *maintains* an estimate of:
- The map M_t (what's where)
- The pose ξ̂_t (where am I)
- The uncertainty σ_t (how confident am I)

In a real system this would be RTAB-Map; in our simulator we have a *proxy* that emulates RTAB-Map's dominant behavior.

### "3D Occupancy Map M_t"
A 3D grid where each cell is one of:
- 0 (free) — observed and confirmed as empty
- 1 (occupied) — observed and confirmed as obstacle
- 0.5 (unknown) — never observed
The "occupancy" terminology comes from probabilistic robotics: each cell stores the probability it's occupied (we use hard 0/0.5/1).

### "Pose Estimate (x̂_t, ŷ_t, ẑ_t)"
The SLAM system's best guess of the UAV's position:
- x̂, ŷ: horizontal position
- ẑ: altitude
The hat (ˆ) indicates *estimate* vs. truth. In the original paper (and old PPT slide) ψ was also included (yaw angle), but we dropped it because we don't store yaw — it's recomputed from velocity direction when needed.

### "Uncertainty surrogate σ_t ∈ [0, 4]"
A scalar that proxies for Tr(Σ_t) (the trace of the pose covariance matrix). Updated each step as:
σ_{t+1} = min(σ_t + 0.03 * |Δp_t| + 0.15 * 𝟙[alt change], 4)
Multiplied by 0.70 on each loop closure. Range [0, 4] is a design choice: 0 = perfectly confident; 4 = maximally uncertain (cap).

### "Observation o_t = (M_t, v_t)"
What the policy sees. Two parts:
- **Map tensor M_t ∈ [0,1]^(5×48×48)**: 5 stacked 48×48 channels (current-altitude occupancy, current-altitude visit mask, trajectory, least-explored altitude occupancy, cross-altitude visit mask)
- **Scalar vector v_t ∈ [0,1]^10**: 10 numbers (normalized σ, frontier count, total coverage, frontier distance, frontier direction, altitude index, current-altitude coverage, best-other-altitude potential, worst-other-altitude coverage, episode step fraction)

### "RL Policy" (Panel C)
The neural network that maps o_t → action distribution. The full architecture is shown in Slide 6.

### "Action a_t = (dx, dy, dz) ∈ [-1, 1]³"
Three continuous values. Interpreted as:
- dx, dy: horizontal velocity command (scaled by step size 3.0)
- dz: altitude vote (thresholded at 0.3)

### "Perception-action loop"
The cycle: sense → process → decide → act → repeat. One iteration = one timestep. Closes the loop because the action changes what the next sensor reading will be.

## Likely Questions

**Q: Why model SLAM as a "proxy" instead of running real SLAM?**
A: Computational cost. A real SLAM stack like RTAB-Map runs at maybe 5-10 Hz and consumes significant CPU/GPU. Training RL takes hundreds of thousands of environment steps; running real SLAM in the loop would multiply training time by 100×. Our proxy captures the essential dynamics (drift accumulation + loop-closure-induced reduction) at almost zero cost, so we can iterate on RL design quickly. Deploying with real RTAB-Map is future work.

**Q: How accurate is the proxy compared to real SLAM?**
A: The proxy is *qualitatively* faithful: σ grows monotonically with travel, drops on loop closures, and is bounded. It's *not quantitatively* the same as RTAB-Map output. The Δσ reward we propose is *agnostic* to the exact dynamics — replace the proxy with real Tr(Σ) and the same shaping principle works.

**Q: Why a 2D LiDAR for a 3D environment?**
A: Because we discretize altitude into 4 distinct layers, each layer is effectively 2D from the agent's perspective. We use a 2D LiDAR per layer plus a half-range "sensor bleed" sweep on adjacent layers (one above, one below) to give the agent a hint about neighboring altitudes. This is much cheaper than 3D point cloud processing and matches what a small UAV can carry.

**Q: What does "perception-action loop" mean exactly?**
A: At each timestep t: (1) the LiDAR fires, (2) the SLAM proxy updates M_t and σ_t based on the LiDAR returns, (3) we construct observation o_t from these, (4) the policy outputs a_t, (5) the env applies a_t to update the UAV's true position, (6) t becomes t+1, repeat. The "loop" closes because the agent's action affects what it senses next.

---

# Slide 5 — Observation and Perception Loop

## What's on the Slide

- A 3D point cloud image (left) labelled "3D point cloud visualization generated by a LiDAR sensor scan"
- Bullet points (right):
  - The system operates on a quadrotor with 360° LiDAR sweep (36 beams)
  - **Observation**: 5-channel map tensor (48×48×5) + 10-dimensional scalar vector
  - **Backend**: RTAB-Map style SLAM maintaining occupancy and pose covariance
  - **Space**: Warehouse partitioned into four distinct altitude layers (1-4m)

> ⚠ Two terms on this slide need correction to match the corrected paper:
> 1. "RTAB-Map style" → should be **"RTAB-Map-inspired"** (signals it's a proxy, not real RTAB-Map)
> 2. "occupancy and pose covariance" → should be **"occupancy and a scalar uncertainty surrogate σ_t"** (we use σ_t, not a full covariance matrix)

## Term-by-Term Breakdown

### "3D point cloud"
A 3D point cloud is a collection of (x, y, z) points returned by a LiDAR. Each point is a single distance measurement projected into 3D space. The image shows what such a point cloud might look like — distant returns in dark blue, near returns in bright green, individual hit points visible against the void of unscanned space.
We don't actually generate a 3D point cloud in our work — we use 2D LiDAR per altitude — but the image is a visual cue for what real LiDAR output looks like.

### "Quadrotor"
A four-rotor UAV. Two pairs of rotors spinning in opposite directions cancel torque; differential thrust controls roll/pitch/yaw/throttle. Examples: DJI Mavic, Parrot AR.Drone, Crazyflie. Our work doesn't depend on a specific quadrotor — any vehicle with a 2D LiDAR and altitude control will do.

### "360° LiDAR sweep"
The LiDAR rotates a full circle and emits laser pulses at regular angular intervals.

### "36 beams"
At 360°/36 = 10° angular resolution. This is *coarse* compared to commercial 2D LiDARs (typically 0.5° resolution = 720 beams), but adequate for our 48×48 grid resolution. Code: `for ray_idx in range(36): angle = 2π × ray_idx / 36`.

### "5-channel map tensor (48×48×5)"
A 3D tensor of shape (5, 48, 48) — 5 separate 48×48 grids stacked. Each channel is one feature about the current altitude:

1. **Current-altitude occupancy** — the SLAM occupancy estimate at the current altitude (free=0, unknown=0.5, occupied=1)
2. **Current-altitude binary visit mask** — 1 wherever the UAV has visited within a 7×7 window of that cell, 0 elsewhere
3. **Exponentially-decaying trajectory trail** — value 1 at the current cell, multiplied by 0.95 each step, leaving a fading trail behind the UAV
4. **Least-explored-altitude occupancy** — the occupancy of the *worst-covered* altitude layer (a "soft prior" hinting where the agent should consider going)
5. **Cross-altitude mean visit mask** — average of the visit masks across all 4 altitudes, showing what's been covered globally

The reason for 5 channels (and not just 1 or 3) is to give the policy multiple complementary spatial views without forcing it to compute these features itself.

### "10-dimensional scalar vector"
A 1-D vector of 10 numbers, all in [0, 1]:

1. **Normalized covariance trace** = σ_t / 4 (normalized to [0,1])
2. **Frontier count** (# frontier cells found) / 50
3. **Total coverage** (fraction of all free cells observed)
4. **Nearest frontier distance** / 24 (24 = grid_size/2)
5. **Relative frontier direction** — angle to the nearest frontier in body frame, mapped to [0,1]
6. **Altitude index** = current_alt / 3 (4 altitudes, normalized)
7. **Current-altitude coverage** (fraction of free cells at this altitude observed)
8. **Best-other-altitude potential** = max over other altitudes of (1 - coverage_at_that_altitude) — captures how much *room to grow* exists elsewhere
9. **Worst-other-altitude coverage** = min over other altitudes of coverage — captures the most-neglected altitude's progress
10. **Episode step fraction** = t / T_max (how far through the episode we are)

These 10 scalars give the policy a compact summary of global state info that's not in the 5-channel map.

### "Backend: RTAB-Map style SLAM"
We say "RTAB-Map style" but the corrected paper says **"RTAB-Map-inspired SLAM proxy"**. Reason: we don't actually run RTAB-Map. We have a *proxy* that mimics RTAB-Map's behavior. RTAB-Map (Real-Time Appearance-Based Mapping) is a popular open-source SLAM system that:
- Runs in real-time
- Uses graph-based pose optimization
- Detects loop closures via feature matching
- Produces dense 3D occupancy grids

Our proxy maintains M_t, ξ̂_t, σ_t with simple update rules — none of the heavy nonlinear least-squares optimization a real SLAM would do.

### "Occupancy"
The map estimate. Each cell stores its occupancy probability. In our case: discrete (free=0, unknown=0.5, occupied=1).

### "Pose covariance"
The covariance matrix Σ_t describing uncertainty in the pose estimate. For a 6-DOF pose (x, y, z, roll, pitch, yaw), Σ is 6×6. The trace Tr(Σ) is a single scalar summary. **In our work, we don't maintain a full Σ — we use a scalar surrogate σ_t directly.** The slide is a slight oversimplification.

### "Warehouse partitioned into four distinct altitude layers (1-4m)"
The warehouse is conceptually divided into 4 horizontal slices at altitudes 1, 2, 3, and 4 meters. Each slice is a 48×48 grid representing the floor plan at that height. The "true" 3D occupancy is a 48×48×4 binary tensor. The UAV occupies exactly one altitude at a time; sensor returns at altitude h fill in the 48×48 grid for that altitude.

## Likely Questions

**Q: Why 5 channels and not just 1 (occupancy)?**
A: Each channel encodes a different prior or feature that helps the policy. The visit mask tells "where have I been"; the trajectory trail tells "what direction was I going recently"; the least-explored-altitude occupancy tells "where do I need to focus"; the cross-altitude visit mask tells "what's been covered globally". All useful, all easy to compute, so we provide them as input rather than make the policy learn to extract them from raw occupancy.

**Q: Why 10 scalars?**
A: Because some quantities are global summaries — they don't fit naturally into a 2D spatial map. The covariance σ is one number describing pose uncertainty; the frontier count is one number; episode progress is one number. These 10 scalars complement the spatial map by giving the policy *non-spatial* state information.

**Q: Why is the LiDAR resolution only 36 beams?**
A: We chose 36 to match the spatial resolution of the 48×48 grid — a 36-beam sweep at 14-cell range covers the local 28-cell-radius circle adequately at our grid resolution. Higher beam count (e.g., 360) would give finer angular resolution but be wasted at the discrete grid level. It's a deliberate design choice for sample efficiency.

**Q: What does "RTAB-Map style" mean and why use a proxy?**
A: "Style" / "inspired" means we mimic the *behavior* of RTAB-Map without running the actual code. We model: drift accumulation as σ grows with travel, brief σ reduction on loop closure, occupancy update from LiDAR returns. We *don't* model: feature extraction, graph pose optimization, full point-cloud handling. Our proxy is fast (microseconds per step) so RL training is feasible. Replacing it with real RTAB-Map is future work.

**Q: Are altitudes physical or symbolic?**
A: They're physical heights (1, 2, 3, 4 meters). On a real UAV, you'd command the height controller to maintain altitude h. In our simulator, we discretize the world into 4 horizontal slices and update only the slice corresponding to current altitude.

---

# Slide 6 — Network Architecture

## What's on the Slide

The CoordConv-LSTM Recurrent PPO Architecture diagram with stages labeled INPUT → ENCODERS → FUSION → MEMORY → HEADS → OUTPUT.

Visible elements:
- **INPUT** (gray boxes):
  - Map Tensor 5×48×48 (5-ch occupancy & priors)
  - Scalars ℝ¹⁰ (10 global features)
- **ENCODERS** (blue boxes):
  - CoordConv2D c=32, s=2 (position-aware convolution)
  - Conv2D c=64, s=2 (deeper features)
  - Conv2D c=64, s=2 (high-level abstraction)
  - Conv2D c=64 + AvgPool₆ + Flatten (flatten to ℝ²³⁰⁴)
  - LayerNorm → FC-64 → FC-32 (scalar embed ℝ³²)
- **FUSION** (purple): Concat + MLP → ℝ²⁵⁶ (fuse modalities)
- **MEMORY** (green): LSTM h_t ∈ ℝ¹²⁸ (temporal memory), with dashed self-loop labeled h_{t-1}
- **HEADS** (orange):
  - Actor FC-256-256 (stochastic policy)
  - Critic FC-256-256 (value estimate)
- **OUTPUT** (red):
  - a_t ~ π(a_t|o_t), μ, σ ∈ ℝ³ (next action)
  - V(o_t) (PPO baseline)

## Term-by-Term Breakdown

### "CoordConv-LSTM Recurrent PPO Architecture"
The full architecture name. Three components:
- **CoordConv**: a CNN with extra coordinate channels for position-awareness
- **LSTM**: Long Short-Term Memory recurrent layer for episode memory
- **Recurrent PPO**: the training algorithm (Proximal Policy Optimization with an LSTM in the policy)

### "INPUT" stage
Two parallel inputs feed the network: the map tensor and the scalar vector. They're processed by separate encoders, then fused.

### "Map Tensor 5×48×48"
The spatial input. 5 channels of 48×48 grids. See Slide 5 for what each channel means.

### "5-ch occupancy & priors"
The 5 channels are *occupancy* (current and least-explored altitude) plus *priors* (visit mask, trajectory, cross-altitude visit). "Priors" because they encode prior information about where the UAV has been.

### "Scalars ℝ¹⁰"
The 10-dimensional non-spatial input. See Slide 5 for what each dimension means. ℝ¹⁰ is mathematical notation for "10-dimensional real vector".

### "10 global features"
The 10 scalars encode *global* (not per-cell) state info: total coverage, frontier count, σ, etc. "Global" distinguishes them from the per-cell *spatial* map features.

### "ENCODERS"
The pair of subnetworks that compress the raw inputs into smaller, learned feature vectors.

### "CoordConv2D c=32, s=2"
- **CoordConv2D**: 2D convolution with extra position-encoding channels. Specifically, two extra channels are appended to the input: a normalized x-coordinate map and y-coordinate map. The 5-channel input becomes 7-channel (5 + 2), then convolved.
- **c=32**: 32 output channels (the conv learns 32 different filters)
- **s=2**: stride 2 (output is half the spatial size of input)
- **kernel=5**: 5×5 receptive field per filter

So this layer takes 5×48×48 input and outputs 32×24×24.

### "position-aware convolution"
The defining feature of CoordConv. Plain convolutions are translation-equivariant (same kernel applied at every position produces the same response). CoordConv breaks this by injecting position information, so the network can encode "this is the center of the map" vs. "this is the boundary".

### "Conv2D c=64, s=2"
- **Conv2D**: standard 2D convolution (no coord channels)
- **c=64**: 64 output channels
- **s=2**: stride 2 (halves spatial size)
- **kernel=3**: 3×3 receptive field

This layer takes 32×24×24 → 64×12×12. Two such layers in sequence: 64×12×12 → 64×6×6.

### "deeper features"
The middle conv layers learn intermediate-level features (combinations of edges into corners, corners into structures, etc.). The depth (4 conv layers total) gives the network a large receptive field — by the last layer, each spatial cell "sees" most of the original 48×48 input.

### "high-level abstraction"
The third strided conv produces highly compressed features at 6×6 spatial resolution. Each cell here represents an 8×8 block of the original input.

### "Conv2D c=64 + AvgPool₆ + Flatten"
The final encoding step. Three operations:
1. **Conv2D c=64 (stride 1)**: refines features at 6×6 (no further downsampling)
2. **AvgPool₆**: adaptive average pooling to a 6×6 output (acts as identity here since the input is already 6×6)
3. **Flatten**: reshapes the 64×6×6 tensor into a 1-D vector of length 64×6×6 = **2304**

### "flatten to ℝ²³⁰⁴"
The result is a 2304-dimensional feature vector.

### "LayerNorm → FC-64 → FC-32"
The scalar branch. Three operations applied to the 10-D scalar vector:
1. **LayerNorm**: per-sample normalization (subtract mean, divide by std) across the 10 features. Stabilizes training when scalars have different scales.
2. **FC-64**: linear layer mapping 10 → 64. Followed by ReLU.
3. **FC-32**: linear layer mapping 64 → 32. Followed by ReLU.

### "scalar embed ℝ³²"
The output of the scalar branch — a 32-dimensional embedding.

### "FUSION"
Combining the spatial and scalar features.

### "Concat + MLP ℝ²⁵⁶"
Two operations:
1. **Concat**: concatenate the 2304-D map features with the 32-D scalar features → 2336-D vector
2. **MLP**: a 2-layer MLP that compresses 2336 → 512 → 256

### "fuse modalities"
"Modalities" because the map (visual) and scalars (numeric) are different *types* of information — different "modes" of input. Fusing them combines both into a single representation.

### "MEMORY"

### "LSTM h_t ∈ ℝ¹²⁸"
The recurrent layer.
- **LSTM**: Long Short-Term Memory, a type of recurrent neural network (RNN) with gates that control information flow.
- **h_t ∈ ℝ¹²⁸**: hidden state is a 128-dimensional vector
- **Single layer**: just one LSTM block (no stacking)

The LSTM takes the 256-D fused embedding z_t plus the previous hidden state h_{t-1} and cell state c_{t-1}, and outputs new (h_t, c_t).

### "temporal memory"
The LSTM's job: remember relevant information across the episode's 600 timesteps. Without it, every action is decided from a single frame's worth of input, which is insufficient for multi-step strategies.

### "h_{t-1}" (dashed self-loop)
The recurrent connection — the previous hidden state is fed back as input to the current step. This is what makes the LSTM remember things.

### "HEADS"

### "Actor FC-256-256"
The policy head:
- Takes h_t (128-D) as input
- Two FC layers, each outputting 256-D, with ReLU between
- Final layer outputs 6-D: μ (3 values for action mean) + log σ (3 values for action std)
- The action is sampled as a_t ~ N(μ, exp(2 log σ))

### "stochastic policy"
The action is sampled from a probability distribution (a Gaussian here), not deterministic. Stochasticity enables exploration and is required by PPO.

### "Critic FC-256-256"
The value head:
- Same architecture as actor (two 256-D FC layers)
- Final layer outputs a single scalar: V(o_t), the predicted total future discounted reward

### "value estimate"
The critic's job: estimate "how good is being in this state, on average, under the current policy?" Used to compute advantages for PPO updates.

### "OUTPUT"

### "a_t ~ π(a_t|o_t)"
The action is sampled from the policy distribution, given observation o_t. The notation "~" means "is sampled from".

### "μ, σ ∈ ℝ³"
The action distribution is parameterized by:
- μ ∈ ℝ³: mean vector (3 values, one per action dimension)
- σ ∈ ℝ³: standard deviation vector (3 values)

So the action is a sample from a 3-D diagonal Gaussian.

### "V(o_t)"
The critic's predicted value of the current observation. Used by PPO to compute advantages: A_t = R_t - V(o_t).

### "PPO baseline"
"Baseline" in policy gradient methods refers to a *control variate* — a function (here, V(o_t)) subtracted from the return to reduce gradient variance without biasing the gradient. The critic provides this baseline.

## Likely Questions

**Q: Why CoordConv only on the first layer?**
A: Once the first layer has access to absolute position, downstream layers can encode position-aware features through their learned weights. Adding coord channels to every layer would be redundant and increase parameters without helping.

**Q: What does "stride 2" mean and why do we use it?**
A: Stride 2 means the convolution kernel skips 2 pixels at a time, halving the output spatial size. We use it to downsample the input from 48×48 to 6×6 across 3 strided layers. Reduces parameters in the FC layers and gives the deeper conv layers a larger effective receptive field.

**Q: Why a final stride-1 conv after the strided ones?**
A: It's a feature-refinement layer — same spatial size (6×6) but learns to combine the high-level features into richer ones. Slight performance boost over removing it.

**Q: Why FC-64 → FC-32 for scalars (not FC-64 → FC-64)?**
A: Compressing 64 → 32 forces the scalar embedding to be more compact and combine the 10 input features into 32 useful dimensions. Empirically works better than larger embeddings; the spatial map is the larger source of information.

**Q: What is LayerNorm and why use it on the scalars?**
A: LayerNorm normalizes across the feature dimension of each sample (subtracts mean, divides by std). The 10 scalars have very different scales (covariance 0-4, fractions 0-1, normalized counts 0-1). LayerNorm ensures the FC layer doesn't get dominated by the largest-scale feature.

**Q: Why concat first and then MLP, instead of MLP-then-concat?**
A: This is "early fusion". It lets the fusion MLP learn cross-modal interactions: e.g., "if covariance is high *and* there's a frontier in the bottom-right, prefer this action". Late fusion (separate processing) would force each branch to commit to a decision before merging.

**Q: Why a 256-D fused embedding (not 128 or 512)?**
A: 256 is the standard SB3 default for PPO with a CNN extractor; it balances capacity and overfitting. We didn't sweep this — the standard worked.

**Q: Why a single LSTM layer and not two?**
A: Empirically, deeper LSTMs are harder to train in RL and rarely help. SB3's RecurrentPPO uses single-layer by default. With 600-step episodes and a 128-dim hidden state, capacity is sufficient.

**Q: Why 128 hidden units and not more?**
A: The default in SB3's MultiInputLstmPolicy. We didn't tune. 128 is enough to memorize the relevant episode context.

**Q: Why two separate 256-256 heads (not a shared one)?**
A: The actor and critic optimize different things (action gradient vs. value gradient). Sharing parameters can cause one objective to interfere with the other ("policy/value entanglement"). Separate heads avoid this and is the SB3 default.

**Q: Why parameterize the actor as μ + log σ instead of μ + σ?**
A: σ must be positive. If we output σ directly, we'd need to clip or use a softplus. Outputting log σ and exponentiating gives σ = exp(log σ) > 0 for free. Plus, log σ has better numerical properties (no sharp transitions near 0).

---

# Slide 7 — Action Space & Altitude Logic

## What's on the Slide

Two columns:
- **(1) ACTION SPACE**:
  - Continuous 3D action: a_t = [Δx, Δy, Δz] ∈ [-1, 1]³
  - Δx, Δy: horizontal movement (in the map plane)
  - Δz: altitude vote (up / down)
  - Diagram showing horizontal movement directions
- **(2) ALTITUDE LOGIC (Δz)**:
  - **ALTITUDE INCREASES**: If Δz > τ_z, move to layer alt(t+1) = alt(t) + 1
  - **ALTITUDE DECREASES**: If Δz < -τ_z, move to layer alt(t+1) = alt(t) - 1
  - **NO ALTITUDE CHANGE**: If |Δz| ≤ τ_z, stay in current layer
  - Dwell Rule: After altitude change, UAV must wait D steps. (t - τ_last(t) ≥ D)
  - Discrete altitude layers: alt ∈ {0, 1, 2, ..., L_max}
- Right-side image: cube layers showing altitudes alt=0 (low), alt=l (mid), alt=L_max (high)

## Term-by-Term Breakdown

### "Continuous 3D action"
The action vector has 3 components, each a continuous real number in [-1, +1]. Continuous (vs. discrete) actions are the natural form for velocity/force commands.

### "a_t = [Δx, Δy, Δz] ∈ [-1, 1]³"
- a_t: action at time t
- Δx, Δy: horizontal velocity vote
- Δz: altitude change vote
- ℝ³ would be unbounded; we restrict to [-1, 1]³ to keep everything in a normalized range.

### "Horizontal movement (in the map plane)"
Δx, Δy specify a direction and magnitude in the 2D plane of the current altitude. Multiplied by step size (3 cells) to give actual movement:
proposed_pos = current_pos + (Δx, Δy) × 3.0
The actual move is clipped to the grid bounds and vetoed if it would collide.

### "Δz: altitude vote (up / down)"
Δz is interpreted as a *vote*, not a continuous height delta. The vote becomes a discrete altitude change (or no change) based on thresholds.

### "ALTITUDE INCREASES: If Δz > τ_z, move to layer alt(t+1) = alt(t) + 1"
- τ_z: threshold (also called τ_alt in the paper). Set to 0.3.
- If the policy outputs Δz > 0.3, the UAV moves up one altitude layer (subject to dwell + boundary checks).
- alt(t+1) = alt(t) + 1: the new altitude index is one higher than the current.
- Subject to: alt(t+1) ≤ L_max (can't go above the top layer)

### "ALTITUDE DECREASES: If Δz < -τ_z, move to layer alt(t+1) = alt(t) - 1"
Symmetric: Δz < -0.3 → move down. Subject to: alt(t+1) ≥ 0 (can't go below the bottom layer).

### "NO ALTITUDE CHANGE: If |Δz| ≤ τ_z, stay in current layer"
The "deadband" — when |Δz| ≤ 0.3, the vote is too weak to trigger a change. The UAV stays at the current altitude.

### "Dwell Rule"
After an altitude change, the UAV must wait at least D = 20 steps before changing altitude again. Even if the policy votes for another change, it's vetoed.
Formally: (t - τ_last(t) ≥ D), where τ_last(t) is the timestep of the most recent altitude change. If t - τ_last < D, altitude votes are ignored.

### "Discrete altitude layers: alt ∈ {0, 1, 2, ..., L_max}"
The altitude is an *integer index*, not a continuous height. With 4 altitudes (heights 1, 2, 3, 4 m), the index ranges over {0, 1, 2, 3} where 0 corresponds to height 1m and 3 corresponds to height 4m. L_max = 3 in our setup.

The mapping from index to physical height is:
height = (alt + 1) × 1m

## Likely Questions

**Q: Why discrete altitudes and not continuous z?**
A: Three reasons. (1) The SLAM Z-filter on a real UAV settles slowly; discrete altitudes with dwell match real flight dynamics. (2) Discretization makes the per-altitude breadth bonus well-defined. (3) Sample efficiency: a continuous z would require the policy to learn fine-grained altitude control, but the actual occupancy grid is discrete in z, so there's no benefit to fine resolution.

**Q: Why threshold = 0.3 specifically?**
A: It's a tuned parameter. With τ_z = 0.3, the deadband is 60% of the [-1, 1] range, meaning the policy must "really want" an altitude change to trigger one. Smaller threshold → too many altitude changes; larger → policy can't change altitude when it should. 0.3 worked across our seeds.

**Q: Why dwell time D = 20?**
A: A real SLAM Z-filter takes about 1 second to settle after an altitude change (typical UAV altitude controller bandwidth). At our timestep (about 50ms in real-time terms), 20 steps ≈ 1 second. So D = 20 corresponds to a realistic Z-filter settle time.

**Q: What if the policy votes for an altitude change but is at the top altitude?**
A: The vote is silently ignored — alt(t+1) = alt(t) = L_max. The policy doesn't get a special signal, but its behavior at the boundary is naturally constrained.

**Q: What is τ_last(t)?**
A: The timestep of the most recent altitude change. Initialized to 0 at episode start. Updated to t every time an altitude change actually occurs.

**Q: Why did you choose to make this a "vote" and not continuous?**
A: The discrete decomposition makes the policy's altitude decision *interpretable* and *trainable*. With a continuous z, training is harder (the policy has to learn fine timing); with a discrete vote with deadband, the policy just decides "up", "down", or "stay" each step. This is a simpler problem to learn.

**Q: Why include Δz in the same continuous action vector instead of as a separate discrete output?**
A: Two reasons. (1) Stable-Baselines3 RecurrentPPO with continuous actions is simpler than a hybrid (continuous + discrete) action space. (2) Putting all three components on the same scale makes the policy gradient natural — same μ, log σ output structure for all three.

**Q: Could the policy abuse the dwell rule to game the reward?**
A: We thought about this. The dwell rule prevents *changes* but doesn't reward staying put. The stagnation penalty (−15 if no per-altitude coverage gain in 25 steps) prevents the policy from staying at one altitude forever and not exploring.

---

# Slide 8 — Reward Design

## What's on the Slide

Multiple components organized by purpose:
- **Objective**: Balance exploration (coverage) and localization accuracy; encourage efficient and informative mapping
- **Exploration Components**: Coverage Gain, Frontier Progress
- **Localization Components**: Uncertainty Reduction, Loop Closure Bonus
- **Altitude-Aware Components**: New Altitude Bonus, Breadth Bonus
- **Penalties**: Collision, Revisit, Stagnation, Altitude Switch Cost

**Total Reward Function** (boxed):
$r_t = w_{\Delta n}\,\Delta n_{\text{cell}} + w_{\Delta d}\,\Delta d_{\text{front}} + w_\sigma\,(\sigma_t/4) + w_{\Delta\sigma}\,\Delta\sigma + w_{\text{alt}}\mathbb{1}[\text{new alt}] + w_{\text{br}}\mathbb{1}[\text{breadth}] + w_{\text{stag}}\mathbb{1}[\text{stag}] + r_{\text{misc}}$

Plus a small drone illustration on the left showing trajectory and discovered cells across 3 layered grids.

## Term-by-Term Breakdown

### "Reward = Coverage gain − Uncertainty + Shaping terms"
The high-level structure: positive reward for coverage, negative reward for uncertainty, plus shaping terms to guide learning.

### "Exploration Components"
Reward terms that directly incentivize covering more of the map.

### "Coverage Gain"
- Term: Δn_cell with weight +1.0
- Number of new cells discovered this step
- Direct, dense, immediate reward for exploration

### "Frontier Progress"
- Term: Δd_front with weight +2.0
- Decrease in distance to nearest frontier this step
- Encourages moving toward unexplored regions
- Clipped to |Δd| ≤ 3 to handle frontier list changes (when a frontier disappears, the distance might jump artificially)

### "Localization Components"
Reward terms that incentivize keeping σ low.

### "Uncertainty Reduction"
- Two terms together: σ_t/4 (with weight -0.8) and Δσ (with weight -0.8)
- σ_t/4 penalizes the *absolute* level of uncertainty
- Δσ penalizes *increases* in uncertainty (and rewards decreases) — our novel contribution
- Together they push the policy toward states where σ is low and trending lower

### "Loop Closure Bonus"
- Term: 𝟙[loop closure] with weight +25
- Discrete reward when a loop closure event fires
- A loop closure dramatically reduces σ (multiplies by 0.70), so the bonus celebrates this rare event

### "Altitude-Aware Components"
Reward terms specifically for the multi-altitude aspect.

### "New Altitude Bonus"
- Term: 𝟙[new altitude] with weight +25
- Paid once per altitude per episode the *first time* the agent visits that layer
- Total possible per episode = 3 × 25 = 75 (the agent starts at altitude 0, so only 3 layers can be "new")

### "Breadth Bonus"
- Term: 𝟙[breadth_a] with weight +60 per altitude
- Paid per altitude when *that* altitude reaches 40% coverage
- Total possible per episode = 4 × 60 = 240
- This is the key term that makes the policy multi-altitude — it back-loads credit onto layers the agent has been neglecting

### "Penalties"
Negative rewards that discourage bad behavior.

### "Collision Penalty"
- Term: 𝟙[collision] with weight -1
- Fires when the agent attempts to move into an occupied cell
- The move is also vetoed (UAV stays put)

### "Revisit Penalty"
- Term: revisit_ratio with weight -0.15
- The "revisit ratio" is the mean of the visit mask in a 7×7 window around the current cell
- If the agent is in heavily-visited territory, this is high; if pushing into new ground, it's low
- Discourages circling

### "Stagnation Penalty"
- Term: 𝟙[stagnation] with weight -15
- Fires when 25 consecutive steps have <2% per-altitude coverage gain
- A heavy penalty to discourage getting stuck

### "Altitude Switch Cost"
- Two terms: 𝟙[alt switch] with weight -3, and 𝟙[alt switch bad] with weight -2
- The first is a flat cost on every altitude change
- The second adds an extra penalty if the agent switches *to* an already-more-explored altitude (i.e., a "wasted" change)

### "Total Reward Function" (the equation)
$r_t = w_{\Delta n}\Delta n_{\text{cell}} + w_{\Delta d}\Delta d_{\text{front}} + w_\sigma(\sigma_t/4) + w_{\Delta\sigma}\Delta\sigma + w_{\text{alt}}\mathbb{1}[\text{new alt}] + w_{\text{br}}\mathbb{1}[\text{breadth}] + w_{\text{stag}}\mathbb{1}[\text{stag}] + r_{\text{misc}}$

This is a *simplification* of the actual 15-term reward; r_misc bundles the smaller terms (collision penalty, per-step cost, revisit penalty, alt switch costs, least-explored bonus, coverage milestone, loop closure bonus). The full table is in our paper as Table II.

### Each weight symbol explained:
- **w_{Δn}**: weight for Δn_cell (= +1.0)
- **w_{Δd}**: weight for Δd_front (= +2.0)
- **w_σ**: weight for absolute σ_t/4 penalty (= -0.8)
- **w_{Δσ}**: weight for delta σ (= -0.8)
- **w_alt**: weight for new altitude bonus (= +25)
- **w_br**: weight for breadth bonus (= +60 per altitude)
- **w_stag**: weight for stagnation penalty (= -15)

### "𝟙[·]" (indicator)
Mathematical notation: returns 1 if the condition inside the brackets is true, 0 otherwise. Used for events that either happen or don't (loop closure, altitude switch, stagnation).

### "Δ" (delta)
The change in a quantity. Δn_cell = number of *new* cells observed this step; Δd_front = *change* in distance to nearest frontier; Δσ = *change* in uncertainty.

## Likely Questions

**Q: How did you choose these specific weights?**
A: Coarse grid search across 3 seeds. We tuned the *relative magnitudes* — sparse-but-large rewards (terminal +300, breadth +60) drive strategic decisions; dense small rewards (per-step -0.05, Δσ -0.8) provide gradient signal; penalties (stagnation -15, terminal σ -120) prevent failure modes. Individual weights weren't fine-tuned; the structure dominates.

**Q: Why is Δσ your novel contribution?**
A: Loop closures are *sparse* — only ~5-10 per episode. That's not enough gradient signal for the policy to learn "actions that lead to loop closures are good". By reframing as Δσ, we get a *dense* per-step signal: every step where σ goes up gets a small penalty; every step where σ drops (mainly loop closure events) gets a large reward. Same information, much denser.

**Q: Why per-altitude breadth bonus instead of all-altitudes-at-once?**
A: Earlier versions of the paper had "+60 when all 4 altitudes simultaneously > 30%". This is a *single* reward event, fragile if seeds differ. Per-altitude (+60 per layer crossing 40%) gives 4 separate reward events, which:
- Stabilizes per-altitude allocation (smaller cross-seed variance — confirmed by ablation)
- Back-loads credit onto the *neglected* altitudes naturally
- Caps total at 240 (still significant)

**Q: What is "revisit_ratio"?**
A: The mean of the binary visit mask in a 7×7 window centered on the current cell. If the UAV is in a region it's already covered thoroughly, the visit mask is mostly 1s in that window → ratio close to 1 → big penalty. If pushing into new ground, ratio close to 0 → no penalty.

**Q: What's the "stagnation" condition exactly?**
A: A counter tracks how long since the last meaningful per-altitude coverage gain. Specifically: starting from when the UAV last entered or arrived at the current altitude, after 25 steps, check whether the cumulative coverage gain at this altitude is <2% (0.02). If so, fire the -15 penalty AND reset the counter.

**Q: Why 40% coverage as the breadth threshold?**
A: 40% is approximately when an altitude has been "meaningfully explored" — past random luck, into deliberate coverage. It's set conservatively: the breadth bonus fires as soon as the policy's effort on that altitude has yielded measurable progress. We tested 30% and 50%; 40% gave the cleanest signal.

**Q: What if the agent never triggers a breadth bonus?**
A: Then it gets no breadth credit, and the terminal coverage bonus is halved (because the breadth-weighting factor ρ in 0.5+0.5ρ stays at 0). The agent loses ~150 reward at episode end if it ignores all altitudes. Strong incentive to spread exploration.

**Q: Why penalize altitude switches?**
A: Without a switch cost, the policy could thrash up/down, gaming the new-altitude bonus repeatedly. The -3 cost, plus the -2 "bad switch" extra, makes thrashing strictly negative. With the costs, the policy switches deliberately, not opportunistically.

**Q: What is the "least-explored altitude" reward?**
A: A small +0.5 per-step bonus given while the UAV is at the least-explored altitude (across all 4). It's a soft pull toward neglected layers — complementary to the breadth bonus (which is sparse and large).

**Q: Why include both σ_t/4 and Δσ?**
A: σ_t/4 keeps the *level* of σ low (penalizes high uncertainty regardless of trend). Δσ keeps the *trend* of σ improving (penalizes increases). Together they encode "be in low-uncertainty states AND keep reducing uncertainty". Either alone is weaker.

---

# Slide 9 — (continuation of Slide 8 reward formula display)

The slide essentially shows the same total reward equation we already broke down. Re-reading it:
$r_t = w_{\Delta n}\,\Delta n_{\text{cell}} + w_{\Delta d}\,\Delta d_{\text{front}} + w_\sigma\,(\sigma_t/4) + w_{\Delta\sigma}\,\Delta\sigma + w_{\text{alt}}\,\mathbb{1}[\text{new alt}] + w_{\text{br}}\,\mathbb{1}[\text{breadth}] + w_{\text{stag}}\,\mathbb{1}[\text{stag}] + r_{\text{misc}}$

Already covered above.

## Likely Questions

**Q: Where do the terminal bonuses appear in this equation?**
A: They're not in r_t — they're added to the *episode-end* reward, separate from per-step. The full per-episode return is:

$\text{Return} = \sum_{t=0}^{T-1} r_t + r_{\text{terminal}}$

where r_terminal = +300 × Cov × (0.5 + 0.5ρ) - 120 × σ_T/4. The terminal covers two things: a big coverage celebration and a final-uncertainty penalty.

**Q: What's r_misc?**
A: A bundle term containing the smaller per-step rewards we didn't write out individually for slide-space reasons:
- 𝟙[collision] × -1
- per-step constant × -0.05
- revisit × -0.15
- 𝟙[alt switch] × -3 (and 𝟙[alt switch bad] × -2)
- 𝟙[at least-explored] × +0.5
- 𝟙[cov ≥ 0.4] × +30 (one-time)
- 𝟙[loop closure] × +25

The full 15-term breakdown is in Table II of the paper.

---

# Slide 10 — Results

## What's on the Slide

Four subfigures:
1. **Top-left**: Final evaluation metrics table across 50 held-out episodes per seed
2. **Top-right**: Coverage vs. step within an evaluation episode (line plot, X = step, Y = coverage %)
3. **Bottom-left**: Learning curve: final coverage vs. training steps
4. **Bottom-right**: Per-altitude learning curves

## Term-by-Term Breakdown

### Top-Left Table

Method names with two performance columns visible (Cov.% and per-altitude breakdowns):
- Random Walk: 55.16
- Nearest Frontier: 52.53
- Spiral: 27.94
- Potential Field: 36.26
- Greedy Info Gain: 82.73 (italicized = best classical)
- RRT Explorer: 39.17
- Vanilla PPO: 86.87 ± 1.19
- Ours (base): 87.56 ± 2.45
- Ours +ΔTr(Σ): 88.69 ± 0.84 ⚠ should be "+Δσ" to match paper
- **Ours full (MS600)**: **91.90 ± 1.72** (bold, headline)
- Abl. no LoopClosure: 89.41 ± 2.31
- Abl. no BreadthBonus: 86.54 ± 1.49

### Each baseline name explained:

**Random Walk**: Action sampled uniformly from [-1, 1]³. Most trivial baseline — what you'd get with no policy. Coverage 55.16%.

**Nearest Frontier**: Yamauchi-style classical method. Pick the closest unexplored frontier; go there; switch altitude when stuck. Coverage 52.53%. Lower than Random Walk because it gets stuck near walls.

**Spiral**: Pre-computed lawnmower waypoints. Visit each in sequence at each altitude. Coverage 27.94% — fails because shelves block the rigid pattern.

**Potential Field**: Virtual repulsion (obstacles), attraction (frontiers), penalty (revisit). Pick the highest-scoring direction. Coverage 36.26% — gets trapped in local minima.

**Greedy Info Gain**: Sample 12 candidate directions; for each, count unknown cells; pick the one revealing most. Coverage 82.73% — strongest classical baseline. But collapses to 69.93% on Alt3.

**RRT Explorer**: Random-tree-based path planner. Sample reachable points; score by neighborhood unknowns; navigate. Coverage 39.17% — RRT struggles on small grids because random sampling rarely hits useful frontiers.

**Vanilla PPO**: Same network, same curriculum, same hyperparameters as ours, but with reward shaping disabled. Only Δn_cell + per-step cost + collision penalty. Coverage 86.87 ± 1.19% — isolates "RL with right architecture" from "RL with our reward design".

**Ours (base)**: Our full reward (15 terms) with T_max=400 (no MS600). Coverage 87.56 ± 2.45%.

**Ours +ΔTr(Σ)** (should be **Ours +Δσ**): Adds the Δσ shaping term to the base. T_max still 400. Coverage 88.69 ± 0.84%. The +1.13 improvement is the contribution of Δσ alone.

**Ours full (MS600)**: Our headline recipe — base + Δσ + extended horizon T_max=600. Coverage 91.90 ± 1.72%. The additional +3.21 is the contribution of horizon extension.

**Abl. no LoopClosure**: Base recipe with the loop-closure reward term removed. Coverage 89.41 ± 2.31% (slightly higher than base, but σ_T rises from 3.71 to 3.82 — uncertainty matters!).

**Abl. no BreadthBonus**: Base recipe with breadth bonus removed. Coverage 86.54 ± 1.49% (-1.02 vs. base). But std on Alt0/Alt3 *doubles* (~5% → ~10%) — breadth bonus stabilizes per-altitude allocation.

### "± 1.72" / "± 2.45"
Standard deviation across the 3 training seeds. We report mean ± std for all RL recipes. Smaller std = more reliable across seeds.

### "Cov. (%)"
Total coverage as a percentage of true free cells observed.

### "Alt0, Alt1, Alt2, Alt3"
Per-altitude coverage at altitudes 1m, 2m, 3m, 4m respectively (Alt0 = altitude index 0 = lowest). Critical to show the *breadth* effect: ours covers all 4 evenly, others collapse on boundary altitudes.

### Top-Right Plot (Coverage vs. Step)

X-axis: episode step (0 to 400). Y-axis: coverage percentage (0 to 80%). Lines for different methods.
- Solid blue: Ours full (MS600) — fastest rise
- Dashed orange: Greedy Info Gain — second fastest
- Green dotted: Nearest Frontier — slow
- Red dashed: RRT Explorer — slowest
- Purple: Vanilla PPO — between Greedy and Ours

The shaded bands are mean ± std across 50 held-out episodes.

The key observation: **Ours hits 50% coverage at step ~180, Greedy at step ~370, Nearest Frontier at step ~580** — we're roughly 2× faster than the best classical baseline.

### Bottom-Left Plot (Learning Curve)

X-axis: training steps (50,000 to ~360,000). Y-axis: final coverage %. Solid blue line with shaded std band, dashed orange horizontal line at 82.7% labeled "Greedy Info Gain baseline".

The plot shows training progression. The policy crosses the Greedy threshold around step 120,000 and continues improving to ~92% by step 200,000+.

### Bottom-Right Plot (Per-Altitude Learning Curves)

X-axis: training steps. Y-axis: per-altitude coverage %. Four lines:
- Alt 0 (RL) — boundary, lowest
- Alt 1 (RL) — middle, top performer
- Alt 2 (RL) — middle, top performer
- Alt 3 (RL) — boundary, lowest

Plus dashed lines showing Greedy's final per-altitude coverage as reference.

The key observation: middle altitudes saturate first (Alt1, Alt2 reach >95% by step 200k), then boundary altitudes catch up (Alt0, Alt3 cross 85% by step 360k). Confirms the multi-altitude strategy.

## Likely Questions

**Q: Why are some baseline coverage values so low (Spiral 27.94%, RRT 39.17%)?**
A: Because the warehouse environment has walls, shelves, and altitude-specific obstacles that defeat simple heuristics. Spiral can't reroute when blocked; RRT's random sampling rarely hits useful frontiers. These methods *would* work in open environments, but real warehouses are cluttered.

**Q: Why does Greedy Info Gain collapse on Alt3?**
A: Alt3 is the topmost altitude with shelf-occluded structures. Greedy maximizes immediate information gain — at altitude 3, the visible space looks small (most of the map is occluded by shelf tops), so Greedy doesn't see a reason to spend time there. It tries other altitudes, gets stuck cycling, and never properly covers Alt3.

**Q: How is your 91.90% statistically different from Vanilla PPO's 86.87%?**
A: Wilcoxon signed-rank pooled across seeds gives p = 7.4×10⁻¹¹ with Cliff's δ = +0.486 (large effect). Even though +5.03 absolute pts looks small numerically, with n=150 paired episodes the difference is highly significant.

**Q: Why is the gap between Ours-base and Ours-MS600 so large (+4.34 pts)?**
A: Two separate improvements: (1) Δσ adds +1.13 (87.56 → 88.69) and (2) extending T_max from 400 to 600 adds +3.21 (88.69 → 91.90). With more steps, the policy can reach the boundary altitudes that take time to cover.

**Q: Why is the std of "Ours base" higher (2.45) than MS600 (1.72)?**
A: Because the base recipe (no Δσ, T_max=400) has less consistent training. The ablation experiments show that without Δσ, training is more seed-dependent. With Δσ + MS600, the policy converges more reliably across seeds.

**Q: What does the per-altitude curves chart tell us?**
A: It shows the *training dynamics* of the multi-altitude strategy. Alt1/Alt2 (middle layers, where shelves are densest) saturate first because they have lots of structure to exploit. Alt0/Alt3 (boundary layers) take longer because they're either floor-level or shelf-top — fewer obvious frontiers. The final policy covers all four >85%, which no classical baseline does.

**Q: What does the time-series plot mean for real-world usefulness?**
A: The 2× speedup to 50% coverage matters because real UAV missions are time-bounded (battery life, mission deadlines, search-and-rescue urgency). A policy that covers 50% in 180 steps vs. 370 is actually useful for early situational awareness.

---

# Slide 11 — Gazebo Simulation

## Current State

⚠ This slide is currently **blank** in the PPT. This is a problem — a blank slide invites the question "what happened here?". Two options:
- **Option A**: Fill it with actual Gazebo transfer results from `gazebo_transfer_eval.py`
- **Option B**: Remove the slide entirely

If you keep it but blank, professors will notice and ask. Below are answers if Option B isn't possible and you must explain a blank slide.

## If Asked About a Blank Slide

**Q: What's on this slide?**
A: We had planned to include preliminary Gazebo transfer experiments here, but we decided to leave them out of this presentation because the Gazebo experiments are work-in-progress — we want to do them properly with the real RTAB-Map back-end before reporting numbers. We list it as future work.

**Q: Have you run any Gazebo experiments?**
A: We have a `gazebo_transfer_eval.py` script that takes our trained policy and runs it inside Gazebo with simulated LiDAR. Initial runs work but the policy was trained on the lightweight grid environment, so transfer is not perfect. We're still tuning the simulator-to-Gazebo transfer; we'll have full numbers in the next iteration.

## Suggested Content if Filling the Slide

If you want to populate this slide, here's what to include:
- A screenshot of the Gazebo simulator with the UAV
- Preliminary numbers: e.g., "Coverage in Gazebo: ~70% (vs 91.90% on lightweight)"
- A diagram showing how the policy transfers: train on lightweight → eval in Gazebo
- Bullet points on transfer challenges: realistic LiDAR noise, real altitude controller, 3D physics
- Acknowledge the gap: "Lightweight-to-Gazebo gap is X pts; further sim-to-real work needed"

---

# Slide 12 — Future Work & Conclusion

## What's on the Slide

**Future Work**:
- Extend to larger and more complex environments
- Increase number of altitude layers / finer 3D resolution
- Develop multi-UAV (multi-agent) coordination
- Improve real-world deployment with onboard sensors
- Integrate advanced perception (LiDAR + vision fusion)

**Conclusion**:
- Learning-based multi-altitude planning significantly improves SLAM efficiency and robustness
- Jointly solves: Where to move, When to change altitude, How to reduce uncertainty
- Key strengths: End-to-end RL policy (CoordConv + LSTM + PPO), Effective reward design for exploration + localization

## Term-by-Term Breakdown

### "Larger and more complex environments"
Currently 48×48×4 (≈48m × 48m × 4 layers). Real warehouses are bigger (200m × 200m+) and more cluttered. Scaling means:
- Larger grid (96×96 or higher)
- More obstacles
- Multi-room layouts with corridors

### "Increase number of altitude layers / finer 3D resolution"
Currently 4 distinct altitudes. Could increase to 8 or 16, possibly use continuous z. Finer resolution means:
- More fine-grained altitude decisions
- More nuanced multi-altitude strategy

### "Multi-UAV (multi-agent) coordination"
Multiple UAVs cooperating to map the environment together. Adds:
- A shared cross-UAV observation channel (each UAV sees what others have mapped)
- Coordination rewards (don't overlap, don't waste effort)
- Possibly a mixed-cooperation/competition framing

### "Real-world deployment with onboard sensors"
Moving from simulation to a physical UAV with:
- Real Velodyne or Ouster LiDAR
- Real RTAB-Map back-end (not a proxy)
- Real flight controller with altitude PID and dwell
- Real-time sensor noise

### "Advanced perception (LiDAR + vision fusion)"
Adding cameras to the LiDAR. Camera advantages:
- Texture and color
- Better feature detection for visual loop closures
- Object recognition (people, vehicles)

LiDAR + vision fusion is the modern SLAM standard.

### "Learning-based multi-altitude planning"
Our overall approach. "Learning-based" because we use RL (no hand-crafted altitude policy). "Multi-altitude" because we treat altitude as a learned discrete decision. "Planning" because the agent decides what to do.

### "SLAM efficiency"
How fast we cover the environment. Measured by time-to-coverage. Our 50% coverage at step 180 vs. Greedy's at step 370 is a 2× efficiency gain.

### "SLAM robustness"
How reliably we achieve good coverage across different maps and conditions. Robustness is what enables the *low standard deviation* we report — only 1.72% std on coverage means our method works *every time*.

### "End-to-end RL policy"
"End-to-end" means the entire pipeline is learned from raw observations to actions, no hand-crafted intermediate stages. Contrast with classical pipelines (frontier extraction → cost map → A* path planning) which has 3 separate hand-crafted modules.

### "CoordConv + LSTM + PPO"
The three key building blocks of our architecture:
- **CoordConv**: position-aware convolution (slide 6)
- **LSTM**: temporal memory (slide 6)
- **PPO**: training algorithm (slide 6)

### "Effective reward design"
Refers to our 15-term composite reward, especially the novel Δσ uncertainty-shaping term and per-altitude breadth bonus. The ablations show each component contributes.

## Likely Questions

**Q: Why is multi-UAV coordination future work and not this paper?**
A: Multi-agent RL is fundamentally harder — you have to handle non-stationarity (each agent's policy changes affect others), coordination protocols, and a much larger joint action space. Single-agent multi-altitude is itself a non-trivial problem; multi-agent doubles the complexity. Best to nail single-agent first.

**Q: Could your method work in outdoor environments?**
A: With modifications. Outdoor has different obstacle distributions, often more open space, and different SLAM challenges (dynamic objects, weather, sun glare on cameras). The framework (CoordConv-LSTM, Δσ shaping, breadth bonus) generalizes; but training distributions and reward weights would need re-tuning.

**Q: What sensors would you add?**
A: Beyond 2D LiDAR: (1) a forward-facing depth camera for fine-grained close obstacles, (2) an IMU for high-rate motion sensing, (3) a downward-facing camera for visual ground-tracking. The IMU is essentially mandatory for any real UAV; LiDAR + camera + IMU is the modern SLAM standard.

**Q: How do you summarize your contribution in one sentence?**
A: We show that a single recurrent policy can jointly decide horizontal movement and discrete altitude changes for active SLAM, achieving 91.90 ± 1.72% coverage with statistically significant large-effect gains over six classical baselines and a learned baseline.

**Q: What was the hardest part of this project?**
A: Designing the reward function. Getting 15 terms to cooperate without one overwhelming the others required ablations and careful balance. The Δσ contribution wasn't obvious initially — we tried other uncertainty signals (raw σ, sigma squared, info gain) before settling on the delta term.

**Q: What surprised you?**
A: How effective the breadth bonus is at *stabilizing* per-altitude allocation, not just raising the mean. Removing it drops mean coverage by only 1 pt, but doubles the std on boundary altitudes — meaning some seeds barely visit Alt0 or Alt3 without that explicit incentive.

**Q: How is your work different from Yin et al.'s MA-SLAM?**
A: MA-SLAM contributes a structured map representation for ground-robot DRL exploration — orthogonal to our work. We focus on the policy and reward design for *multi-altitude UAV active SLAM*, where altitude is the new decision variable.

**Q: How is your work different from NavRL?**
A: NavRL trains PPO for safe 3D UAV flight with sim-to-real transfer, but the objective is collision-free *navigation* (point A to point B) — not active mapping. We target a fundamentally different task.

**Q: Where would you go next with this?**
A: Three priorities: (1) replace the σ proxy with real RTAB-Map output to validate the Δσ shaping survives realistic SLAM dynamics, (2) run Gazebo transfer experiments to measure the lightweight-to-physics gap, (3) extend to multi-agent. Approximately in that order.

---

# Slide 13 — Thank You

A standard thank-you slide. No technical content.

## Anticipated Wrap-Up Questions

**Q: How long did this project take?**
A: Approximately 3-4 months of active work (one academic semester). About 1-2 months on research and design, 1 month on implementation and training, 1 month on evaluation and writing.

**Q: How would you scale this to a final-year project / thesis?**
A: Three extensions: (1) real Gazebo or hardware experiments, (2) multi-agent coordination, (3) larger / more diverse environments. Each could be a 6-month effort.

**Q: Are you planning to publish?**
A: Yes — we're targeting ICAAV 2026 (or a comparable IEEE/Springer venue) for a 4–6 page conference paper. The full 6-page IEEEtran version is ready; we'd need to convert to Springer Nature format for ICAAV submission.

**Q: What did you each contribute?**
A: (Tailor to your actual division of work — e.g., one of you focused on env + reward design, the other on architecture + training; both reviewed and wrote.)

**Q: What did you learn?**
A: Practical lessons in (1) reward engineering — small changes have big effects, (2) statistical rigor — Wilcoxon + Cliff's δ vs. just mean is critical, (3) ablation discipline — every claim needs a controlled experiment, (4) reproducibility — code and paper must align exactly.

---

# Appendix — Quick Reference Sheet for the Day of Presentation

## Numbers to memorize

| Quantity | Value |
|---|---|
| Headline coverage | **91.90 ± 1.72%** |
| Best classical (Greedy) | 82.73% |
| Vanilla PPO | 86.87 ± 1.19% |
| Gap over Greedy | +9.17 pts |
| Gap over Vanilla | +5.03 pts |
| Time to 50% (Ours) | ~180 steps |
| Time to 50% (Greedy) | ~370 steps |
| Loop closures (Ours full) | 7.49 ± 1.17 |
| Loop closures (Greedy) | 0.36 |

## Architecture: 32 → 64 → 64 → 64
## Channels in input map: 5 (occupancy, visit mask, trajectory, least-explored, cross-altitude)
## Scalars: 10 (σ, frontier count, total cov, frontier dist, frontier dir, alt idx, current alt cov, best other potential, worst other cov, episode step)
## LSTM hidden: 128
## Heads: (256, 256)
## Action: [-1, 1]³ continuous (dx, dy, dz)
## Altitude threshold: τ_alt = 0.3
## Dwell: D = 20 steps
## Training: 800,000 steps
## γ = 0.995, GAE λ = 0.97, clip = 0.2, batch 128

## All 7 Wilcoxon p-values: < 10⁻¹⁰
## All Cliff's δ values: > +0.48 (all "large" effect by convention)

## If asked ONE thing to emphasize:

> "Our key contribution is the Δσ reward term, which densifies the otherwise sparse loop-closure signal. Combined with a per-altitude breadth bonus that stabilizes layer allocation, we achieve 91.90% coverage with statistically significant large-effect gains over six classical baselines and a learned baseline — all on a single recurrent policy that jointly decides horizontal movement and discrete altitude changes."

That's the elevator pitch. Lead with it.

---

# End of Document

*Read in order. Drill the Q&A sections. Memorize the numbers in the Quick Reference. You're ready.*
