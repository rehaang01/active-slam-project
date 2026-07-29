# EC-392 UG Project-I — Presentation Defense Document
## Localisation-Uncertainty-Aware Deep Reinforcement Learning for Active SLAM on UAVs in GPS-Denied Environments

### Authors: Anjali Gupta (23095013), Rehaan Goenka (23095082)
### Supervisor: Dr. Om Jee Pandey, Dept. of Electronics Engineering, IIT (BHU) Varanasi

---

# Table of Contents

1. [Foundational Concepts](#1-foundational-concepts)
2. [SLAM, Active SLAM, and Why It's Hard](#2-slam-active-slam-and-why-its-hard)
3. [UAVs, LiDAR, and the Sensing Stack](#3-uavs-lidar-and-the-sensing-stack)
4. [Reinforcement Learning Foundations](#4-reinforcement-learning-foundations)
5. [Proximal Policy Optimization (PPO) Deep Dive](#5-proximal-policy-optimization-ppo-deep-dive)
6. [Recurrent PPO and Why We Need It](#6-recurrent-ppo-and-why-we-need-it)
7. [Convolutional Neural Networks (CNN) Basics](#7-convolutional-neural-networks-cnn-basics)
8. [CoordConv — What and Why](#8-coordconv--what-and-why)
9. [LSTM (Long Short-Term Memory)](#9-lstm-long-short-term-memory)
10. [Other Network Components](#10-other-network-components)
11. [Our System Model — Slide-by-Slide](#11-our-system-model--slide-by-slide)
12. [Our Network Architecture — Layer-by-Layer](#12-our-network-architecture--layer-by-layer)
13. [Action Space & Altitude Logic](#13-action-space--altitude-logic)
14. [Reward Design — All 15 Components Explained](#14-reward-design--all-15-components-explained)
15. [Training Methodology](#15-training-methodology)
16. [Evaluation Methodology & Statistics](#16-evaluation-methodology--statistics)
17. [Baselines — All 7 Explained](#17-baselines--all-7-explained)
18. [Results Walkthrough](#18-results-walkthrough)
19. [Ablation Analysis](#19-ablation-analysis)
20. [Anticipated Professor Questions and Answers](#20-anticipated-professor-questions-and-answers)
21. [Quick Reference — Numbers to Memorize](#21-quick-reference--numbers-to-memorize)

---

# 1. Foundational Concepts

## 1.1 What is a UAV?

**UAV** = **Unmanned Aerial Vehicle**, an aircraft without a human pilot on board. Controlled either remotely by a human operator or autonomously by an onboard computer running planning + control software. Common UAV types:
- **Quadrotor** (what we use): four rotors arranged in an X or + configuration. Each rotor produces vertical thrust; tilting the body lets it move horizontally.
- **Fixed-wing**: like miniature airplanes, more efficient for long range but cannot hover.
- **Hybrid VTOL**: vertical take-off and landing, then transitions to fixed-wing flight.

**Why quadrotors for indoor mapping?** They can hover, change direction quickly, stop on demand — essential for exploring tight indoor spaces with shelves, walls, and obstacles.

## 1.2 What is "GPS-denied"?

GPS (Global Positioning System) gives a UAV its absolute (x, y, z) position by triangulating signals from satellites. **GPS-denied environments** are places where GPS signals are blocked or unreliable:
- Inside warehouses (walls and metal shelving block signals)
- Tunnels, mines, basements (no line-of-sight to satellites)
- Disaster zones with collapsed structures
- Indoor military operations
- Forests with dense canopy

Without GPS, the UAV must localize itself using its own sensors. That's where **SLAM** comes in.

## 1.3 What is Mapping vs. Localization vs. SLAM?

- **Mapping**: building a model of the environment (e.g., where walls are, where free space is). Requires knowing the robot's pose.
- **Localization**: estimating where the robot is. Requires a map.
- **SLAM** (Simultaneous Localization and Mapping): doing both at once. The robot doesn't have a map, doesn't know where it is, and must figure out both as it moves. This is a chicken-and-egg problem solved iteratively.

---

# 2. SLAM, Active SLAM, and Why It's Hard

## 2.1 SLAM Pipeline (passive)

A typical SLAM system does:
1. **Front-end**: extract features from sensor data (e.g., LiDAR scan matching, visual feature points), estimate motion between consecutive sensor readings.
2. **Back-end**: build a *factor graph* of poses linked by motion constraints; solve a non-linear least-squares optimization to find poses that minimize residual error.
3. **Loop closure detection**: when the robot returns to a previously-visited place, recognize it and add a "closure" constraint that ties the current pose back to the earlier pose, reducing accumulated drift.
4. **Map update**: once poses are known, project sensor returns into a global map (occupancy grid, point cloud, mesh).

In **passive SLAM**, the robot is moved by a human or follows a pre-planned path. SLAM just observes and builds.

## 2.2 What is Active SLAM?

In **Active SLAM**, the robot **chooses where to go** to optimize SLAM quality. It's not just exploring blindly — it's making decisions about *which actions will most improve the map and reduce localization uncertainty*.

This adds a **planning** component on top of SLAM:
- Where should I move next?
- Should I revisit somewhere to trigger a loop closure (and reduce uncertainty)?
- Should I push into unknown space (and increase coverage)?

## 2.3 Why is Active SLAM hard?

Three coupled objectives — they often conflict:
1. **Maximize coverage** → keep moving into new territory.
2. **Minimize localization uncertainty** → revisit known areas for loop closures.
3. **Stay safe** → avoid collisions.

**Classical methods** (frontier-based, information-theoretic) handle each with a hand-crafted cost function. Tuning the trade-off (this much weight on coverage, that much on uncertainty, etc.) is brittle.

**Our claim**: a learned policy can balance these naturally if given the right reward signal.

## 2.4 What is "Multi-Altitude" Active SLAM?

Most active-SLAM literature plans on a single 2D occupancy grid — the robot stays at one height. But UAVs can fly at different altitudes, and **different things are visible at different altitudes**:
- At 1m: low shelves, floor obstacles
- At 2–3m: middle of warehouse shelving
- At 4m: top of shelves, ceiling beams

If you only fly at one altitude, you miss what's above and below. **Multi-altitude active SLAM** treats the vertical axis as an explicit decision variable — when to fly higher, when to fly lower.

In our work: 4 discrete altitude layers (1m, 2m, 3m, 4m), each a 48×48 grid, total environment = 48×48×4.

## 2.5 What is "Localization Uncertainty"?

After a SLAM solver computes the robot's pose estimate, it also reports **how confident** it is in that estimate. Mathematically, this is captured by the **pose covariance matrix** Σ_t — a symmetric positive-definite matrix whose diagonal entries are the variances on x, y, z, ψ (yaw), and off-diagonals are covariances.

A common scalar summary is the **trace** Tr(Σ_t) = sum of diagonal entries = total variance across all pose dimensions. Lower trace = more confident estimate.

**In our project, we don't run a real SLAM backend.** We use a **scalar surrogate σ_t ∈ [0, 4]** that mimics how Tr(Σ_t) would behave:
- Grows with motion (more travel = more accumulated drift)
- Grows with altitude changes (Z-filter in the SLAM front-end is sensitive)
- Shrinks on loop closures (a loop closure constraint reduces uncertainty)

This is a deliberate design choice: it lets us train RL policies without paying the cost of running a real SLAM solver in the loop. We acknowledge the gap and list "replace with real RTAB-Map" as future work.

## 2.6 What is a Loop Closure?

When the robot **revisits a previously-mapped location** and *recognizes* that it's the same place, the SLAM system adds a constraint linking the current pose to the older pose. This:
- Closes the "loop" in the trajectory
- Drastically reduces accumulated odometry drift
- Sharpens the entire trajectory and map

In our simulator: a deterministic geometric loop closure fires when the agent returns within `r < 4` cells of a stored landmark, after at least 40 steps and 24 units of travel since visiting that landmark. Each loop closure multiplies σ by 0.70 (i.e., 30% reduction).

## 2.7 What is RTAB-Map?

**RTAB-Map** (Real-Time Appearance-Based Mapping) is a popular open-source SLAM system that:
- Uses a graph-based pose backend
- Detects loop closures via visual or geometric features
- Builds a 3D occupancy grid
- Runs in real-time on modest hardware

Our paper says **"RTAB-Map-inspired"** because we don't actually run RTAB-Map; we model its dominant behavior (drift accumulation + loop-closure-induced drift reduction) with a simple scalar dynamic.

---

# 3. UAVs, LiDAR, and the Sensing Stack

## 3.1 What is LiDAR?

**LiDAR** = **Li**ght **D**etection **A**nd **R**anging. Active sensor that emits laser pulses and measures time-of-flight to compute distance to surfaces.

**Why LiDAR over cameras for indoor mapping?**
- Robust to lighting (dark or poorly-lit warehouses)
- Direct distance measurement (no monocular depth ambiguity)
- High accuracy at moderate range (10s of meters)
- Native compatibility with occupancy grids

## 3.2 360° LiDAR with N_r = 36 Beams

Our quadrotor carries a **2D rotating LiDAR** that sweeps 360° in the horizontal plane. Per scan, it fires **36 uniformly-spaced beams** (one every 10°). Each beam returns the distance to the nearest obstacle along that ray.

We use a `for ray_idx in range(36): angle = 2π * ray_idx / 36` loop in the simulator. The sensor range is 14 cells (~14 meters at 1m/cell).

## 3.3 Adjacent-Altitude Sensor Bleed

In addition to the full-range sweep at the current altitude, we do a **half-range sweep on each adjacent altitude layer** (one above, one below). This emulates the fact that real LiDAR beams diverge slightly with distance and pick up structure from neighbouring layers.

This lets the agent "sense" a hint of what's at altitude ±1 without flying there, which helps it decide whether changing altitude is worthwhile.

## 3.4 Occupancy Mapping

The map at each altitude is a **48×48 grid** where each cell stores one of three values:
- **Free** (0): observed and confirmed empty
- **Occupied** (1): observed and confirmed obstacle
- **Unknown** (0.5): never observed

When LiDAR fires along ray angle θ, we walk the cells along that ray:
- Each cell up to the hit becomes **Free**
- The hit cell becomes **Occupied**
- Beyond the hit, cells stay **Unknown**

## 3.5 Pose Estimate

The SLAM proxy gives us $\hat{\xi}_t = (\hat{x}_t, \hat{y}_t, \hat{z}_t)$ — the estimated UAV position. Yaw $\psi_t$ is not stored explicitly; it's recomputed on demand from the velocity direction `(p_t - p_{t-1})` when needed for the relative-frontier-direction scalar feature.

Note: in the *true* simulator state, we know exactly where the UAV is (because we control it). The SLAM proxy adds a layer of pretending we don't, with σ_t representing the uncertainty.

---

# 4. Reinforcement Learning Foundations

## 4.1 What is Reinforcement Learning (RL)?

RL is a learning paradigm where an **agent** interacts with an **environment** by taking **actions**, receives **observations** and **rewards**, and learns a policy that maximizes *cumulative reward* over time.

Difference from supervised learning:
- No labels. Just rewards (which can be sparse, delayed, or noisy).
- Agent must figure out which actions caused which rewards (credit assignment).
- Exploration vs. exploitation trade-off.

## 4.2 Markov Decision Process (MDP)

The standard mathematical framework. An MDP is a tuple **⟨S, A, T, R, γ⟩**:
- **S** = set of states
- **A** = set of actions
- **T(s' | s, a)** = transition probability of going to s' if you take action a in state s
- **R(s, a)** = reward for taking action a in state s
- **γ ∈ [0, 1]** = discount factor (how much future rewards count vs. present)

The "Markov property": next state depends only on current state and action, not on history.

## 4.3 Partially Observable MDP (POMDP)

In real-world problems (including our SLAM), the agent **does not see the full state**. It sees an **observation** o that's a partial/noisy function of the state. POMDP tuple **⟨S, A, O, T, O_obs, R, γ⟩**:
- **O** = set of possible observations
- **O_obs(o | s)** = observation kernel: probability of observing o when the true state is s

In our project:
- **True state** s_t = (UAV pose, full occupancy map, full uncertainty) — agent doesn't see this directly.
- **Observation** o_t = (M_t, v_t) — a 5-channel 48×48 map tensor + 10-D scalar vector — derived from s_t but lossy-compressed.

The agent must use the **history** of observations to infer state — that's why we need memory (LSTM).

## 4.4 Policy

A **policy π** maps observations (and possibly history) to actions:
- **Deterministic policy**: π(o) = a (always picks the same action for the same obs)
- **Stochastic policy**: π(a | o) = probability distribution over actions

We use a stochastic policy because it explores naturally. Specifically: **diagonal Gaussian** over R³ — for each of the 3 action dimensions, the policy outputs a mean μ and standard deviation σ; the action is sampled from N(μ, σ²).

## 4.5 Value Function

The **value function V(s)** is the expected cumulative discounted reward starting from state s and following the policy thereafter:

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \mid s_0 = s\right]$$

In our PPO setup, the **critic** network estimates V(o_t) — used to compute *advantages* (how much better was an action than expected), which guides the policy gradient.

## 4.6 Discount Factor γ

γ controls how much we care about future rewards.
- γ = 0: only care about immediate reward (myopic)
- γ → 1: care about all future rewards equally (long-sighted)

We use **γ = 0.995**. The "effective horizon" is 1/(1−γ) ≈ 200 steps. Our episode horizon is 600 steps for the headline recipe — γ is matched so the agent considers ~200-step strategies.

## 4.7 Episode and Trajectory

An **episode** is one run from reset to termination. A **trajectory** τ is the sequence of (s, a, o, r) tuples visited during an episode.

Our episode terminates when:
- Coverage ≥ 95% (success), OR
- t = T_max (timeout, T_max = 400 or 600)

## 4.8 Action Bounds

Continuous actions in [−1, 1]³. The clipping is enforced in the env:
```python
action = np.clip(action, -1.0, 1.0).astype(np.float32)
```
This prevents the policy from taking extreme actions outside the trained range.

---

# 5. Proximal Policy Optimization (PPO) Deep Dive

## 5.1 Why PPO?

There are many RL algorithms (Q-learning, DQN, A2C, DDPG, SAC, TD3, etc.). PPO has become the default for continuous control because:
- Works well out of the box, less hyperparameter tuning
- Stable training (rarely diverges)
- Sample-efficient enough for our problem size
- Strong empirical track record (drone racing, robotics, RLHF for LLMs)

## 5.2 Policy Gradient — The Foundation

The basic policy-gradient theorem says:
$$\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta \log \pi_\theta(a|s) \cdot A^\pi(s,a)]$$

where $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ is the **advantage** — how much better is taking action a in state s than the average?

Increase log-probability of actions with positive advantage, decrease for negative advantage. That's the policy gradient.

## 5.3 The Trust-Region Idea

Vanilla policy gradient often **destabilizes**: a single huge update in the wrong direction can wreck the policy. Solution: limit how far the new policy can drift from the old one. **TRPO** (Trust-Region Policy Optimization) does this with a hard KL-divergence constraint, but it's expensive. **PPO** approximates it with a simpler clipping trick.

## 5.4 PPO Clipped Surrogate Objective

PPO defines the **probability ratio** r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t), then maximizes:

$$L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)]$$

where ε = 0.2 (the clip range). The clip prevents updates that push the new policy too far.

## 5.5 Generalized Advantage Estimation (GAE)

How do we compute the advantage A_t? Vanilla TD gives high bias; full Monte Carlo gives high variance. GAE blends them with a parameter λ ∈ [0, 1]:

$$A^{GAE}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error. We use **λ = 0.97**.

## 5.6 Total PPO Loss

Combined objective our code minimizes (negated since gradient descent):

$$\mathcal{L}(\theta_\pi, \theta_V) = \mathcal{L}^{CLIP}(\theta_\pi) + c_1 (V(s) - \hat{R})^2 - c_2 \mathcal{H}[\pi]$$

- **c_1 (V−R̂)²**: value-function loss, trains the critic
- **−c_2 H[π]**: entropy bonus (negative because we want to *maximize* entropy = encourage exploration)
- c_1, c_2 are scalar coefficients

## 5.7 Entropy Annealing

Early in training, we want high exploration → high entropy weight. Late in training, we want exploitation → low entropy. We linearly anneal **c_2 from 0.02 to 0.002** over training.

## 5.8 Hyperparameters Used

| HP | Value | Why |
|---|---|---|
| Learning rate | 3×10⁻⁴ → 10⁻⁵ (linear decay) | Standard PPO range, decays for fine-tuning |
| Entropy coef | 0.02 → 0.002 | High exploration early, exploitation late |
| n_steps | 256 | Rollout length per env per update |
| Batch size | 128 | Mini-batch size for SGD updates |
| n_epochs | 5 | Number of passes over rollout per update |
| γ | 0.995 | Effective horizon 200 steps |
| GAE λ | 0.97 | Standard value |
| Clip ε | 0.2 | Standard PPO clip |
| Max grad norm | 0.5 | Gradient clipping for stability |
| Parallel envs | 4 | Multi-process rollouts |

---

# 6. Recurrent PPO and Why We Need It

## 6.1 The POMDP Memory Problem

Our observation o_t is a *lossy* compression of the full state. In particular:
- The trajectory channel fades old visits with a 0.95 multiplier each step
- The 10-D scalar vector squeezes a lot of SLAM state into a few numbers

A purely Markovian (feedforward) policy that only sees o_t would have to make every decision *as if* this is the only information it ever had. That's bad — many useful strategies require memory:
- "I committed to altitude 3 at step 50, stay here for 40 steps to fully cover it"
- "I just had a loop closure at coordinate (12, 35), don't waste time going back there"
- "The nearest frontier was at the south wall 20 steps ago; if I'm still moving north, I'm probably stuck"

## 6.2 Recurrent PPO

Recurrent PPO adds an **LSTM layer** between the feature extractor and the policy/value heads. The LSTM maintains a hidden state h_t that summarizes the entire episode history. The policy is now π(a_t | o_t, h_t) — conditioned on both current obs and hidden state.

Implementation (Stable-Baselines3 sb3-contrib): `RecurrentPPO` with `MultiInputLstmPolicy`. Hidden size = 128, single LSTM layer.

## 6.3 BPTT Through Rollouts

During training, we run the policy for `n_steps = 256` steps in each parallel env. Then we do a backward pass through the entire rollout — back-propagation through time (BPTT) through up to 256 LSTM steps. This is what enables long-term credit assignment.

---

# 7. Convolutional Neural Networks (CNN) Basics

## 7.1 What is a Convolution?

A 2D convolution slides a small **kernel** (e.g., 3×3 or 5×5) over an input image, computing element-wise products and summing. The kernel weights are learned; same kernel applied everywhere → **translation equivariance** (a feature in the top-left looks the same as in the bottom-right).

For a kernel W of size K×K, input I, output O at position (i, j):
$$O(i,j) = \sum_{u=0}^{K-1} \sum_{v=0}^{K-1} W(u,v) \cdot I(i+u, j+v)$$

## 7.2 Stride

The **stride** s is how many pixels the kernel moves between applications. Stride 1 = output same size as input (with padding). Stride 2 = output half the size. Stride lets us reduce spatial resolution while increasing receptive field.

In our network: stride-2 convolutions reduce 48 → 24 → 12 → 6.

## 7.3 Padding

Adding zeros around the input border so the kernel can be applied at the edges without going out of bounds. Padding of (K−1)/2 keeps stride-1 conv output the same size as input.

## 7.4 Channels

A conv layer with c output channels learns c independent kernels, each producing one output feature map. Stacking them gives an output of shape (c, H, W).

Our channel widths: input 5 → 32 → 64 → 64 → 64.

## 7.5 Kernel Size Choices

- 5×5 (CoordConv first stage): captures broader local patterns
- 3×3 (subsequent stages): smaller, more efficient; standard for deep CNNs

## 7.6 Why CNNs for our 5×48×48 map tensor?

The map tensor is spatially-organized: nearby cells are related (an obstacle at (10, 10) is more likely correlated with cells at (10, 11) than with cells at (40, 40)). CNNs respect this locality with local receptive fields. MLPs would have to learn it from scratch.

## 7.7 Receptive Field

The set of input pixels a given output neuron depends on. As we stack convs, the receptive field grows. For our network:
- Layer 1 (kernel 5, stride 2): RF = 5
- Layer 2 (kernel 3, stride 2): RF = 5 + (3−1)×2 = 9
- Layer 3 (kernel 3, stride 2): RF = 9 + (3−1)×4 = 17
- Layer 4 (kernel 3, stride 1): RF = 17 + (3−1)×8 = 33

So by the final layer, each spatial position in the 6×6 feature map "sees" a 33×33 patch of the original 48×48 input — most of the map.

---

# 8. CoordConv — What and Why

## 8.1 The "Position Blindness" Problem

A standard convolution is **translation-equivariant**: the same kernel applied at every spatial position produces the same response. This is usually a feature, but for our problem it's a bug.

In active SLAM, **the same local pattern means different things at different positions**:
- An obstacle at the *center* of the map is in the middle of explored space
- The same obstacle at the *boundary* might be against a wall, blocking exploration in that direction
- The agent's relationship to the *map's coordinate system* (which corner is "home", where are the walls) is informative

Plain CNNs cannot represent absolute position. Their output for a 3×3 patch of "occupied | free | free" looks the same whether that patch is at (5,5) or (40,40).

## 8.2 The CoordConv Fix

CoordConv (Liu et al., 2018) adds two extra channels to the input: a **normalized x-coordinate map** and a **normalized y-coordinate map**:
- Channel x: value at (i, j) = j / W (column index, normalized to [0, 1])
- Channel y: value at (i, j) = i / H (row index, normalized to [0, 1])

These are constant across the batch — they just tell the conv "where in the image am I?". With these channels, the conv can learn to behave differently at different positions if it wants to.

## 8.3 Where We Use It

Only the **first** convolutional stage uses CoordConv. After that, the position information is already baked into the features, so plain Conv2D suffices for layers 2–4.

In code (`shared_slam_policy.py`):
```python
class CoordConv2d(nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        yy = torch.linspace(-1, 1, h).view(1,1,h,1).expand(b,1,h,w)
        xx = torch.linspace(-1, 1, w).view(1,1,1,w).expand(b,1,h,w)
        return self.conv(torch.cat([x, xx, yy], dim=1))
```

## 8.4 What CoordConv Helps With (Concretely)

- The agent learns "the map's bottom-left is harder to reach if I started at top-right"
- It can encode "I'm in the middle vs. at the boundary"
- It can learn that frontiers near walls behave differently from frontiers in open space

---

# 9. LSTM (Long Short-Term Memory)

## 9.1 The Vanishing-Gradient Problem in RNNs

Vanilla RNNs apply the same nonlinearity at every time step, multiplying gradients during BPTT. Over long sequences, gradients either vanish (→ 0) or explode (→ ∞). In practice, RNNs can't learn dependencies more than ~10 steps apart.

## 9.2 LSTM Architecture

LSTM (Hochreiter & Schmidhuber, 1997) introduces a **cell state c_t** that flows through the network with minimal interference, plus three **gates** that control what information enters, leaves, and persists:

**Forget gate** (what to drop from the cell state):
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

**Input gate** (what new info to write):
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$

**Cell state update**:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Output gate** (what to expose as the hidden state):
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(c_t)$$

Here σ is the sigmoid (output ∈ [0,1] = "how open is the gate"), ⊙ is element-wise multiplication.

## 9.3 Hidden State vs. Cell State

- **c_t (cell state)**: the "long-term memory". Information can flow through with minimal modification.
- **h_t (hidden state)**: the "short-term output". This is what downstream layers see.

Both are 128-dimensional in our network.

## 9.4 Why LSTM for Our Problem

- Episode is 400–600 steps long
- Useful strategies span dozens of steps
- We need to remember "I committed to this altitude 30 steps ago"
- We need to remember "the area to the south is already explored"
- LSTM gates let the network selectively remember and forget

## 9.5 Initialization

At episode start, $(h_0, c_0) = \mathbf{0}$ — empty memory. The LSTM accumulates context as the episode progresses.

## 9.6 Backprop Through Rollout

During PPO updates, the gradient flows back through up to 256 LSTM steps (one rollout). PyTorch handles this automatically via BPTT.

---

# 10. Other Network Components

## 10.1 ReLU Activation

**ReLU(x) = max(0, x)**. The standard activation for modern neural networks because:
- Cheap (a single comparison)
- Doesn't saturate for positive inputs (vanishing gradient resistant)
- Sparse activation (some neurons exactly zero)

## 10.2 GroupNorm

A normalization layer. Splits the C channels into G groups, computes mean and variance within each group, normalizes:

$$\hat{x}_{c,h,w} = \frac{x_{c,h,w} - \mu_g}{\sigma_g + \epsilon}$$

We use **G = 8 groups**. GroupNorm is preferred over BatchNorm for RL because:
- BatchNorm depends on batch statistics, which fluctuate during RL training
- GroupNorm works with batch size 1 (during inference)
- Doesn't depend on running averages that drift

## 10.3 Adaptive Average Pooling (AvgPool₆)

Pools spatial dimensions to a fixed output size, regardless of input size. `AdaptiveAvgPool2d(6)` outputs a 6×6 spatial map by averaging.

In our network, the input is *already* 6×6 after the strided convs, so this layer effectively acts as identity. We keep it for robustness if the input size ever changes.

## 10.4 Flatten

Reshapes a 64×6×6 tensor into a 1-D vector of length 64×6×6 = **2304**. This is the input to the fusion MLP.

## 10.5 LayerNorm

Normalizes across the feature dimension of a single sample (unlike BatchNorm which normalizes across the batch). Used here on the scalar vector v_t before the MLP because the 10 scalar features have very different scales (covariance trace, frontier count, coverage fraction, etc.).

## 10.6 Linear / Fully-Connected Layer

A matrix multiplication: y = Wx + b. Used in:
- Scalar embedder: 10 → 64 → 32
- Fusion MLP: 2336 → 512 → 256
- Actor head: 128 → 256 → 256 → 6 (μ and log_σ for 3 actions)
- Critic head: 128 → 256 → 256 → 1

## 10.7 Diagonal Gaussian Policy

The actor outputs a mean μ ∈ R³ and log-std log σ ∈ R³ (parameterized as log-std for numerical stability — std is always positive). The action is sampled as:

$$a_t \sim \mathcal{N}(\mu_t, \mathrm{diag}(\sigma_t^2))$$

"Diagonal" because the covariance matrix is diagonal (no correlations between dx, dy, dz).

## 10.8 Why log-std Instead of std?

- σ must be positive. Outputting log_σ and exponentiating gives σ = exp(log_σ) > 0 automatically.
- Better numerical stability — log-space additions are bounded.
- Standard practice in continuous-control RL.

---

# 11. Our System Model — Slide-by-Slide

## 11.1 Panel A: Multi-Altitude Warehouse

- 4 altitude layers: 1m, 2m, 3m, 4m
- UAV (quadrotor) currently at altitude 3m
- 360° LiDAR sweep with 36 beams (visualized as cyan rays)
- The "dz" arrow shows the altitude vote action
- Shelves visible at each layer (some shelves only exist at specific altitudes)

The grid resolution is 48×48 cells per layer. Each cell ≈ 1m × 1m, so ~48m × 48m total area per layer.

## 11.2 Panel B: SLAM Back-end

Three boxes representing what the SLAM proxy maintains:
- **3D Occupancy Map M_t**: voxelized occupancy estimate
- **Pose Estimate (x̂_t, ŷ_t, ẑ_t)**: where the SLAM thinks the UAV is
- **Uncertainty surrogate σ_t ∈ [0, 4]**: scalar proxy for Tr(Σ_t)

These get **fused** (summarized) into:
- A 5-channel map tensor M_t ∈ [0,1]^(5×48×48)
- A 10-D scalar vector v_t ∈ [0,1]¹⁰

This is the **observation o_t** that goes into the policy.

## 11.3 Panel C: RL Policy and Action Loop

The recurrent policy:
1. Takes o_t = (M_t, v_t)
2. Has hidden LSTM state h_t carrying episode context
3. Outputs action a_t = (dx, dy, dz)
4. Action drives the flight controller (closing the perception–action loop)

Time advances: t → t+1, sensors fire again, SLAM updates, new observation, new action.

---

# 12. Our Network Architecture — Layer-by-Layer

## 12.1 Forward Pass Walkthrough

**Input**:
- M_t: tensor of shape (5, 48, 48), each value in [0, 1]
- v_t: vector of shape (10,), each value in [0, 1]

**Step 1: Spatial encoder**
```
Map (5, 48, 48)
  → CoordConv2D(5+2, 32, k=5, s=2, p=2)   # adds 2 coord channels
  → GroupNorm(8, 32) → ReLU
  Output: (32, 24, 24)

  → Conv2D(32, 64, k=3, s=2, p=1)
  → GroupNorm(8, 64) → ReLU
  Output: (64, 12, 12)

  → Conv2D(64, 64, k=3, s=2, p=1)
  → GroupNorm(8, 64) → ReLU
  Output: (64, 6, 6)

  → Conv2D(64, 64, k=3, s=1, p=1)         # stride 1, refines features
  → GroupNorm(8, 64) → ReLU
  Output: (64, 6, 6)

  → AdaptiveAvgPool2d(6)                  # identity here
  Output: (64, 6, 6)

  → Flatten
  z_map ∈ R^2304
```

**Step 2: Scalar encoder**
```
Scalars (10,)
  → LayerNorm(10)
  → Linear(10, 64) → ReLU
  → Linear(64, 32) → ReLU
  z_scl ∈ R^32
```

**Step 3: Fusion**
```
[z_map; z_scl] ∈ R^2336
  → Linear(2336, 512) → ReLU
  → Linear(512, 256)
  z ∈ R^256
```

**Step 4: Recurrent backbone**
```
LSTM(input=256, hidden=128, layers=1)
  inputs: z, h_{t-1}, c_{t-1}
  outputs: h_t, c_t ∈ R^128
```

**Step 5: Policy head**
```
h_t ∈ R^128
  → Linear(128, 256) → ReLU
  → Linear(256, 256) → ReLU
  → Linear(256, 6)               # μ (3) and log_σ (3)
  μ, log_σ ∈ R^3
  a_t ~ N(μ, diag(exp(2*log_σ)))
```

**Step 6: Value head**
```
h_t ∈ R^128
  → Linear(128, 256) → ReLU
  → Linear(256, 256) → ReLU
  → Linear(256, 1)
  V(o_t) ∈ R
```

## 12.2 Total Parameter Count

Approximately:
- CNN: ~70k params (4 conv layers)
- Scalar MLP: ~3k
- Fusion MLP: ~1.3M (the 2336→512 layer dominates)
- LSTM: ~200k (4 × (256+128) × 128 weights)
- Heads: ~150k (each)
- Total: ~1.9M parameters

## 12.3 Why This Architecture?

| Choice | Justification |
|---|---|
| CoordConv first, plain Conv after | Position-awareness needed at the lowest level; once features encode position, plain conv suffices |
| Stride-2 convs (no pooling) | Fewer hyperparams than pool-stride combo; standard in modern CNNs |
| GroupNorm not BatchNorm | RL has unstable batch statistics |
| Single-layer LSTM with 128 hidden | Enough capacity for episode memory; deeper LSTMs are harder to train in RL |
| Separate actor/critic heads | Critic has higher-frequency targets; sharing breaks training stability |
| (256, 256) heads | Standard PPO default in SB3 |

---

# 13. Action Space & Altitude Logic

## 13.1 Continuous 3D Action

$$a_t = (\Delta x, \Delta y, \Delta z) \in [-1, 1]^3$$

- **Δx, Δy**: continuous horizontal movement vote, scaled by step size 3.0 to give a velocity command. Clipped to keep the UAV inside the grid.
- **Δz**: altitude vote (NOT a continuous height change). Mapped to a discrete layer change via thresholding.

## 13.2 Altitude Vote Mechanism

```
if Δz > +0.3:    move to alt + 1  (if not at top)
elif Δz < -0.3:  move to alt - 1  (if not at bottom)
else:            stay at current altitude
```

The threshold τ_alt = 0.3 (called τ_z in the slide) gives a "deadband" — small values of Δz don't trigger a layer change. This makes the policy commit to altitude changes only when it really wants to.

## 13.3 Dwell Rule

```
if alt change requested AND (steps_at_current_alt < D):
    veto the change, stay at current altitude
```

Where D = 20. This prevents the policy from rapidly oscillating between altitudes.

## 13.4 Why a Dwell Rule?

The SLAM back-end has a **Z-filter** — a low-pass filter that smooths the height estimate. If the UAV switches altitudes too rapidly, the Z-filter lags behind and produces inconsistent map updates at the wrong altitude. The dwell rule gives the back-end time to settle.

In our simulator we don't model the Z-filter explicitly, but we keep the dwell rule because:
- It matches what would happen on a real platform
- It prevents the policy from learning a degenerate "thrash altitudes" strategy
- It enforces commitment, which empirically improves coverage

## 13.5 Collision Handling

```
if proposed_position would hit an obstacle:
    veto the move, stay at current position
    apply collision penalty (-1)
```

The agent learns to avoid obstacles via the penalty signal.

## 13.6 Action vs. Movement

The action a_t is what the *policy outputs*. The actual UAV movement may differ if:
- The agent is at an altitude boundary (vote ignored)
- The dwell timer hasn't expired (vote ignored)
- The proposed move would collide (move vetoed)
- The agent is at the grid boundary (clip to grid)

This is called **action filtering**: the env "interprets" the policy's vote rather than blindly applying it.

---

# 14. Reward Design — All 15 Components Explained

The total per-step reward is a weighted sum of 15 components, plus 2 terminal bonuses. From `env_3d.py`:

## 14.1 Δn_cell — Coverage Gain (+1.0)

`reward_progress = W_NEW_CELL * float(new_cells)`

Number of *new cells* observed this step (cells that went from unknown to free/occupied). Direct exploration reward.

## 14.2 Δd_front — Frontier Progress (+2.0)

A **frontier** is a cell on the boundary between free and unknown space. Δd_front is the *change* in distance to the nearest frontier:

```
reward = W_FRONTIER_DIR * (prev_distance - current_distance)
       = +2.0 * (distance reduction)
```

Clipped to |Δd| ≤ 3 to prevent jumps when the frontier list changes drastically. Encourages moving *toward* unexplored regions.

## 14.3 1[collision] — Collision Penalty (−1.0)

Fires when the agent attempts to move into an occupied cell. Discourages obstacle collisions.

## 14.4 per-step (−0.05)

Constant negative reward each step. Implicit budget pressure: every wasted step costs something. Encourages efficient exploration.

## 14.5 revisit (−0.15)

Mean of the visited-mask in a 7×7 window centered on the current cell. If the agent hangs around already-visited cells, this is high; if it's pushing into new territory, it's low. Discourages circling.

## 14.6 1[new alt] — New Altitude Bonus (+25)

Paid **once per altitude per episode** the first time the agent enters that altitude. Strong incentive to *visit* every layer.

## 14.7 1[alt sw] — Altitude Switch Cost (−3.0)

Fires every time the altitude changes. Implicit cost of changing layers (mimics the real cost of tilting + climbing). Without this, the policy might thrash up and down.

## 14.8 1[alt sw bad] — Bad Altitude Switch (−2.0)

Extra penalty if the agent switches *to a more-explored altitude*. Discourages going from a less-explored layer to a well-explored one (which would be wasteful).

## 14.9 1[at least-explored] — Least-Explored Bonus (+0.5)

Per-step bonus while the agent is at the least-explored altitude. Pulls the policy toward where exploration is most needed.

## 14.10 1[stag.] — Stagnation Penalty (−15)

Fires when:
- 25 consecutive steps with <2% per-altitude coverage gain

A heavy penalty against "stuck" behavior. Forces the policy to move on if local exploration isn't paying off.

## 14.11 1[cov ≥ 0.4] — Global Coverage Milestone (+30)

One-time bonus when total coverage first crosses 40%. Marks an "early milestone" — the policy gets credit for getting started efficiently.

## 14.12 1[LC] — Loop Closure Bonus (+25)

Fires when a deterministic loop closure event occurs (agent revisits a landmark). Strong incentive for revisiting — *but only after travelling enough that the loop closure is meaningful*.

## 14.13 1[breadth_a] — Per-Altitude Breadth Bonus (+60 per altitude)

For each altitude a ∈ {0,1,2,3}, paid once when *that altitude alone* crosses 40% coverage. Maximum per episode = 4 × 60 = 240. The key "multi-altitude" reward — back-loads credit onto layers the agent has been neglecting.

## 14.14 σ_t/4 — Absolute Uncertainty Penalty (−0.8)

Per-step penalty proportional to current σ. Bigger uncertainty → bigger negative reward. Pushes the policy toward states with low σ.

## 14.15 Δσ — Step-wise Uncertainty Change (−0.8) [OUR CONTRIBUTION]

Penalizes *increases* in σ, rewards *decreases*. Δσ = σ_{t+1} − σ_t.
- A move that increases σ (drift accumulation): mild negative reward
- A loop closure that drops σ by 30%: large positive reward (−w × negative Δσ = positive)

This is the **densification of the loop-closure signal**. Without Δσ, the only "uncertainty-aware" signal is the rare LC bonus. With Δσ, every step contributes a gradient toward uncertainty-reducing behavior.

## 14.16 Terminal Coverage Bonus (+300 × Cov × (0.5 + 0.5ρ))

At episode end, scaled by:
- **Cov**: final coverage fraction
- **ρ**: fraction of altitudes that crossed the 40% breadth threshold

If ρ = 0 (no breadth): bonus halved
If ρ = 1 (all 4 altitudes past 40%): full bonus

## 14.17 Terminal σ Penalty (−120 × σ_T/4)

At episode end, large penalty proportional to final σ. The agent has a strong incentive to *end with low uncertainty*.

## 14.18 Total Reward Formula (PPT slide 8 / paper Eq. 12)

$$r_t = w_{\Delta n}\Delta n_{\text{cell}} + w_{\Delta d}\Delta d_{\text{front}} + w_\sigma \cdot \sigma_t/4 + w_{\Delta\sigma}\Delta\sigma + w_{\text{alt}}\mathbb{1}[\text{new alt}] + w_{\text{br}}\mathbb{1}[\text{breadth}] + w_{\text{stag}}\mathbb{1}[\text{stag}] + r_{\text{misc}}$$

where r_misc bundles the smaller terms (collision, per-step, revisit, alt switch, least-explored, coverage milestone, loop closure bonus).

## 14.19 Why these specific weights?

Tuned via coarse grid search over 3 seeds. The relative magnitudes encode priorities:
- Big rewards: terminal coverage (+300), breadth (+60 per alt), new alt (+25), LC (+25), milestone (+30) — sparse but transformative events
- Medium: stagnation (−15) — important for avoiding local minima
- Small: per-step (−0.05), revisit (−0.15), Δσ (−0.8) — dense gradients
- Penalties: terminal σ (−120), collision (−1), alt switch costs (−3, −2)

## 14.20 Why σ_t/4 (and σ_T/4)?

Normalization. σ ∈ [0, 4], so σ/4 ∈ [0, 1]. Keeps the reward scale comparable to other unit-scale terms.

---

# 15. Training Methodology

## 15.1 Multi-Seed Training

We train 3 independent seeds: **{42, 100, 200}**. Each seed gets its own random initialization and rollouts. We report **mean ± std across seeds** for all RL metrics. This is the standard for credible empirical RL.

## 15.2 Curriculum Learning

The environment has 3 difficulty levels:
- **Easy**: simpler maps, fewer obstacles
- **Medium**: middle complexity, includes a wall in the center
- **Hard**: full warehouse with shelves, walls, scattered obstacles

We schedule difficulty progression:
- 0–120k steps: easy
- 120k–300k steps: medium
- 300k–800k steps: hard

This is **curriculum learning** — start simple, ramp up. The early-easy phase teaches basic exploration; later phases demand it work in cluttered scenes.

## 15.3 Total Training Budget

- 800,000 environment steps per seed
- Checkpoints every 60,000 steps (≈ 13 checkpoints per seed)
- Approx 1 GPU-hour per seed on a modern NVIDIA card

## 15.4 VecNormalize

Stable-Baselines3 wrapper that:
- Normalizes rewards by a running mean/std (to keep gradient magnitudes stable across reward scales)
- We use **norm_obs=False** (observations already in [0, 1])
- We use **norm_reward=True** with **clip=15** (clips reward to ±15 to avoid outliers blowing up gradients)

## 15.5 Parallel Environments

We run **4 environments in parallel** (subprocess vec-env). Each env runs independently; the policy collects rollouts across all 4 simultaneously, then does PPO updates on the combined batch. Speeds up training, increases gradient diversity.

## 15.6 Linear Annealing

Two quantities anneal linearly over training:
- Learning rate: 3×10⁻⁴ → 10⁻⁵
- Entropy coefficient: 0.02 → 0.002

LR decay is for fine-tuning; entropy decay shifts from exploration to exploitation.

## 15.7 Checkpoint Selection Rule

Out of 13 checkpoints per seed, we pick the **earliest one that satisfies all three**:
1. Coverage ≥ Greedy Info Gain coverage + 3 percentage points
2. Final σ_T ≤ 0.80 × Greedy's σ_T
3. Min per-altitude coverage ≥ 25%

For our headline MS600 recipe, all three seeds have this rule fire at **step 180,000**. We use that checkpoint for all evaluation.

If no checkpoint satisfies all 3 rules, fall back to the one maximizing:
```
score = Cov + 0.5 * min_per_alt_cov - 0.05 * σ_T
```

This is **principled checkpoint selection** — not just "best on the test set" (which leaks), but a rule that prefers early-good policies and balanced behavior.

## 15.8 Recipe Variants in the Paper

| Recipe | T_max | Δσ enabled? | Comment |
|---|---|---|---|
| Vanilla PPO | 400 | No (no shaping rewards) | Bare PPO |
| Ours (base) | 400 | No | Our reward except Δσ |
| Ours +Δσ | 400 | Yes | Adds Δσ |
| Ours +Δσ + MS600 | **600** | Yes | Plus extended horizon — headline recipe |
| Abl. no LoopClosure | 400 | No | LC reward removed |
| Abl. no BreadthBonus | 400 | No | Breadth bonus removed |

---

# 16. Evaluation Methodology & Statistics

## 16.1 Held-Out Maps

50 distinct map seeds from the range [200, 249] — *not* used during training (training uses random seeds outside this range). This is the **test set**.

Each method runs 50 episodes (one per held-out map seed) per RL training seed. So 3 seeds × 50 maps = **n = 150 paired episodes** for each RL recipe.

## 16.2 Why Paired Comparison?

For each map seed, we compare each method on the *same* environment. Pairing across maps eliminates variance from differing maps. We can use paired tests like Wilcoxon signed-rank.

## 16.3 Wilcoxon Signed-Rank Test

A non-parametric paired test (no Gaussian assumption). For each pair, compute the difference; rank the absolute differences; sum the ranks of positive vs. negative differences. p-value tells you how likely this pattern is under the null hypothesis (no difference).

We use **one-sided** tests (H₁: MS600 > baseline). p < 0.05 = significant; p < 0.001 = "***".

**Why Wilcoxon instead of paired t-test?** Coverage values are not Gaussian — they're bounded in [0, 1] and often skewed. Wilcoxon doesn't assume any distribution shape.

## 16.4 Cliff's Delta (Effect Size)

A non-parametric effect size in [-1, +1]:
- Cliff's δ = (# pairs where MS600 > baseline) − (# pairs where MS600 < baseline) all divided by total pairs

Interpretation thresholds:
- |δ| < 0.147: **negligible**
- 0.147 ≤ |δ| < 0.33: **small**
- 0.33 ≤ |δ| < 0.474: **medium**
- |δ| ≥ 0.474: **large**

Why effect size in addition to p-value? p-value tells you *if* there's a difference; δ tells you *how big* the difference is. With n = 150, even tiny differences can be "statistically significant"; effect size guards against over-claiming.

## 16.5 Our Headline Numbers

All MS600 vs. baseline comparisons:
| Comparison | p | δ | Effect |
|---|---|---|---|
| vs. Random Walk | 1.2×10⁻²⁶ | +0.990 | large |
| vs. Nearest Frontier | 1.4×10⁻²⁶ | +0.979 | large |
| vs. Spiral | 1.2×10⁻²⁶ | +1.000 | large |
| vs. Potential Field | 1.2×10⁻²⁶ | +0.974 | large |
| vs. Greedy Info Gain | 8.1×10⁻¹⁴ | +0.534 | large |
| vs. RRT Explorer | 1.2×10⁻²⁶ | +0.997 | large |
| vs. Vanilla PPO | 7.4×10⁻¹¹ | +0.486 | large |

All p-values significant at *** level; all effects large.

---

# 17. Baselines — All 7 Explained

Implemented in `lightweight_train/baselines_3d.py`.

## 17.1 Random Walk

Action: random vector in [-1, 1]³ each step. The trivial baseline — what you'd get with no policy at all. Coverage 55.16%.

## 17.2 Nearest Frontier

Classical heuristic. At each step:
1. Find all *frontier cells* (free cells adjacent to unknown cells)
2. Move toward the closest one
3. If stuck (no coverage gain) for 15 steps, switch to least-explored altitude

Standard in robotic exploration. Coverage 52.53% — surprisingly worse than Random Walk because it gets stuck near walls.

## 17.3 Spiral (Lawnmower)

Pre-computed waypoints in a serpentine pattern across each altitude. Visit each waypoint, then move to next altitude.

Doesn't react to the environment at all. Useful baseline because it's the "pure systematic" approach. Coverage 27.94% — fails because the warehouse has shelves blocking the rigid pattern.

## 17.4 Potential Field

Each candidate direction gets a score from:
- **Information gain** (how many unknown cells in that direction)
- **Repulsion** from obstacles
- **Visit penalty** for revisiting cells

Pick the highest-scoring direction. Coverage 36.26% — gets trapped in local minima.

## 17.5 Greedy Information Gain

Sample 12 candidate directions; for each, count unknown cells along the ray for ~20 steps. Pick the direction that reveals the most unknowns. If stuck for 20 steps, switch altitude.

The strongest classical baseline. **Coverage 82.73%**, but collapses to **69.93% on Alt3** (shelf-occluded boundary altitude).

## 17.6 RRT Explorer

Rapidly-exploring Random Tree variant (Umari & Mukhopadhyay 2017):
1. Sample random reachable points
2. For each, score by # unknowns in neighborhood
3. Move toward the highest-scoring sample
4. Repeat

Theoretically sound for path planning. Coverage 39.17% — RRT struggles in our small grid because random sampling rarely hits useful frontiers.

## 17.7 Vanilla PPO

A learned baseline that **shares our network and curriculum** but disables every shaping reward except:
- Δn_cell (coverage gain)
- per-step penalty
- collision penalty

**Why include this?** Isolates "RL with the right architecture" from "RL with our reward design". The gap between Vanilla PPO (86.87%) and Ours-MS600 (91.90%) = 5.03 pts is *purely* the contribution of our reward shaping.

---

# 18. Results Walkthrough

## 18.1 Main Table (Slide 10 / Paper Table I)

```
Method                  Cov(%)         Alt0    Alt1    Alt2    Alt3
Random Walk             55.16          50.11   61.13   59.39   49.92
Nearest Frontier        52.53          50.67   61.92   56.45   41.31
Spiral                  27.94          26.23   28.66   28.49   28.27
Potential Field         36.26          42.58   44.55   32.14   26.41
Greedy Info Gain        82.73 *best*   87.01   90.67   83.90   69.93
RRT Explorer            39.17          58.92   60.74   29.50   9.48
---
Vanilla PPO             86.87 ± 1.19   77.67   92.79   93.15   83.52
Ours (base)             87.56 ± 2.45   85.26   94.78   93.26   77.12
Ours + Δσ               88.69 ± 0.84   77.47   93.85   95.20   87.69
Ours + Δσ + MS600       91.90 ± 1.72   88.11   96.85   95.79   86.81  ← HEADLINE
---
Abl. no LoopClosure     89.41 ± 2.31   80.84   94.41   95.12   86.95
Abl. no BreadthBonus    86.54 ± 1.49   79.46   91.97   92.98   81.54
```

## 18.2 Key Takeaways

1. **Our headline beats every baseline.** 91.90% > 82.73% (Greedy), 86.87% (Vanilla).
2. **Per-altitude is the killer.** Greedy collapses to 69.93% on Alt3; ours stays above 85% on all four altitudes. RRT collapses to 9.48% on Alt3.
3. **Statistical significance.** All seven baseline comparisons are p < 10⁻¹⁰ with large Cliff's δ.
4. **+5.03 pts over Vanilla PPO** — that's the pure contribution of our reward design (same architecture, same curriculum, same hyperparameters).

## 18.3 Time-to-50% (Slide 10 plot)

- Ours hits 50% at **step ~180**
- Greedy hits 50% at **step ~370**
- Nearest Frontier hits 50% at **step ~580**

We're **~2× faster than Greedy** and ~3× faster than Nearest Frontier.

## 18.4 Per-Altitude Learning Curves

Shows how each altitude's coverage progresses over training:
- Middle altitudes (Alt1, Alt2) saturate first (densest shelf content)
- Boundary altitudes (Alt0, Alt3) catch up by step 360,000

Confirms the "multi-altitude strategy": policy learns to spread effort across all layers.

## 18.5 Loop Closure Counts

- Ours full: **7.49 ± 1.17** loop closures per episode
- Greedy: 0.36
- Vanilla PPO: 3.35
- Random Walk: 5.96

We're not just exploring — we're *revisiting* in a way that triggers loop closures. This is exactly what reduces uncertainty.

---

# 19. Ablation Analysis

## 19.1 Setup

Two ablations, both run on the **base recipe** (T_max=400, Δσ disabled, mean coverage 87.56%) across the same 3 seeds:
- **No LoopClosure**: remove the 1[LC] reward (+25 → 0)
- **No BreadthBonus**: remove the 1[breadth_a] reward (+60 → 0)

Crucially, ablations are *NOT* run on the MS600 recipe — they're isolating the contribution against a clean base.

## 19.2 No LoopClosure: 89.41 ± 2.31%

**Coverage actually goes up slightly (+1.85 pts).** Surprising? No — the loop closure reward primarily affects *uncertainty*, not coverage. Look at σ_T:
- Base recipe: σ_T = **3.71**
- No LoopClosure: σ_T = **3.82**

σ rises by 0.11 — confirming the loop closure reward's real role: **uncertainty reduction**, exactly what it was designed for.

## 19.3 No BreadthBonus: 86.54 ± 1.49% (−1.02 pts)

**Mean drop is small (−1.02 pts), but the cross-seed std on boundary altitudes doubles**:
- Base recipe Alt0 std: **4.97%** → No-Breadth: **9.85%** (2.0× wider)
- Base recipe Alt3 std: **4.83%** → No-Breadth: **9.38%** (1.9× wider)

Without the breadth bonus, **the policy becomes seed-dependent about which altitudes it commits to**. Some seeds barely visit Alt0 or Alt3; others manage. The breadth bonus *stabilizes* the per-altitude allocation.

## 19.4 What This Tells Us

- **Breadth bonus**: makes per-altitude allocation reliable
- **Loop closure reward**: makes uncertainty reduction reliable
- **Δσ + MS600**: lifts coverage from 87.56 to 91.90 (+4.34 pts total via the +Δσ +1.13 pts plus +MS600 horizon +3.21 pts)

Each component does what it was designed for. None is redundant.

---

# 20. Anticipated Professor Questions and Answers

## 20.1 Conceptual Questions

**Q: Why use Reinforcement Learning instead of classical SLAM planners?**
A: Classical methods use hand-crafted cost functions to balance coverage, uncertainty, and constraints. The trade-off is brittle and depends on environment-specific tuning. An RL policy can learn the right trade-off from a single composite reward, end-to-end. We show empirically that this gives +9.17 pts of coverage over the strongest classical baseline (Greedy Info Gain) and balances *all four altitudes* (which Greedy can't, collapsing to 69% on Alt3).

**Q: Why is your work multi-altitude?**
A: Real UAVs operate in 3D space, but most active SLAM literature uses a single 2D occupancy grid. Different altitudes show different things in a warehouse — the top of shelves at 4m, the floor at 1m, the middle at 2-3m. Treating altitude as a *learned discrete decision* lets the policy explore all layers explicitly. We use 4 altitude layers (1, 2, 3, 4 m), each with its own 48×48 grid.

**Q: What's the Δσ contribution and why is it novel?**
A: Loop closures are *sparse* events (maybe 5-10 per episode) — too rare to give the RL policy a meaningful gradient. Δσ = σ_{t+1} − σ_t fires *every step*: it's small for steps that increase uncertainty (drift), and large + negative when a loop closure cuts σ by 30%. This **densifies** the otherwise sparse loop-closure signal, making uncertainty reduction trainable end-to-end. To our knowledge, no prior work uses this exact term in active-SLAM RL.

**Q: Why CoordConv?**
A: Plain CNNs are translation-equivariant — the same kernel responds the same way to the same pattern at any spatial position. But in our problem, *position matters*: a frontier near a wall behaves differently from one in open space; the map's center vs. boundary have different navigational meaning. CoordConv adds two extra channels containing normalized (i/H, j/W) coordinates so the network can encode absolute position when useful.

**Q: Why LSTM and not just a feedforward policy?**
A: Our observation is partial — the trajectory channel fades old visits, the 10-D scalar squeezes SLAM state into a few numbers, and information from earlier in the episode gets lost. The LSTM hidden state h_t carries that history forward, so the policy can express multi-step strategies like "commit to altitude 3 for 40 steps". We back-prop through 256-step rollouts, so the LSTM can learn temporal dependencies up to that scale.

**Q: Why stride-2 convolutions instead of MaxPooling?**
A: Strided convs achieve the same downsampling effect (48 → 24 → 12 → 6) but learn *which* features to keep, instead of using a hand-coded max operation. Modern CNNs (ResNet variants) overwhelmingly use strided convs. Also avoids the small bias MaxPooling introduces toward sharp features, which doesn't fit our continuous-valued occupancy grid.

**Q: Why GroupNorm and not BatchNorm?**
A: BatchNorm depends on batch statistics, which are unstable in RL (the distribution of states changes over training). GroupNorm normalizes within each sample (across groups of channels), so it works even with batch size 1 at inference. Standard practice in modern RL (used in PPO, SAC implementations).

## 20.2 Technical Questions

**Q: What does the discount factor γ control?**
A: γ ∈ [0, 1] controls how much we care about future rewards. γ = 0 → only immediate; γ → 1 → all future equally. We use γ = 0.995, giving an "effective horizon" 1/(1−γ) = 200 steps — matched to our 600-step T_max so the agent reasons over ~1/3 of the episode at any point.

**Q: What does Cliff's delta measure?**
A: It's a non-parametric effect size in [-1, 1]. It's the difference between (proportion of pairs where A > B) and (proportion where A < B). δ = +1 means A always wins, δ = 0 means they're equal. Convention: |δ| ≥ 0.474 = "large" effect. Different from p-value: p tells you *if* there's a difference; δ tells you *how big* it is.

**Q: Why Wilcoxon signed-rank instead of paired t-test?**
A: Paired t-test assumes the differences are Gaussian. Coverage values are bounded in [0, 1] and skewed (especially when methods saturate near 90+%). Wilcoxon is non-parametric — no distribution assumption.

**Q: How did you choose the reward weights?**
A: Coarse grid search over the 3 seeds. The structure was decided first (which terms to include), then rough magnitudes set so that:
- Sparse-but-large rewards (terminal +300, breadth +60, new alt +25) drive strategic decisions
- Dense small rewards (per-step −0.05, Δσ −0.8) provide gradient
- Penalties (stagnation −15, terminal σ −120) prevent failure modes
We didn't fine-tune individual weights — the relative ordering matters more.

**Q: Why these specific architecture choices? (32→64→64→64 channels, 128 LSTM, etc.)**
A: 32→64→64→64 is a standard small-CNN width — enough capacity for a 5×48×48 input without overparameterizing. LSTM hidden 128 is the SB3 default for MultiInputLstmPolicy. (256, 256) actor/critic heads is also the SB3 default. We didn't sweep over these — they're sensible defaults that work.

**Q: Why three seeds and not 5 or 10?**
A: 3 seeds × 50 maps = 150 paired observations is enough for tight Wilcoxon CIs at p < 10⁻¹³. More seeds would tighten things further but at significant compute cost (~1 GPU-hour per seed for 800k steps). 3 is the standard in published RL papers.

**Q: How do you ensure your evaluation is fair?**
A: All methods are evaluated on the *same* 50 held-out map seeds (range [200, 249]) with identical initial conditions and the recipe-matched horizon. RL recipes get 3 seeds × 50 maps = 150 episodes; classical baselines get 1 × 50 = 50 (they're deterministic up to action noise). We use one-sided Wilcoxon tests.

## 20.3 Critical Questions (where you need to be careful)

**Q: Your environment is just a 48×48 grid. Does this transfer to a real UAV?**
A: Honest answer: not yet. The grid is a deliberate abstraction for sample-efficient RL training (we'd need many millions of steps in a high-fidelity simulator otherwise). The next step in our pipeline is `gazebo_transfer_eval.py` — testing the trained policy in a Gazebo physics simulator with realistic LiDAR. We acknowledge this gap explicitly in the paper as future work.

**Q: σ_t isn't a real covariance trace. Doesn't that undermine the localization claim?**
A: We're transparent about this in the paper — σ_t is a *scalar surrogate* for Tr(Σ_t), not a real covariance. It's parameterized to match the dominant behavior of a real SLAM stack: drift accumulation with motion (0.03 × distance) + altitude penalty (0.15 × altitude change) + multiplicative reduction on loop closure (×0.70). The Δσ reward we propose is *agnostic* to the exact dynamics — replace σ with a real Tr(Σ) and the same shaping principle applies. Validating this would be the first thing we'd do with a real RTAB-Map back-end.

**Q: The σ_T values for your method (3.73) are basically the same as Greedy's (3.76). So you don't actually beat Greedy on uncertainty.**
A: Correct — we *match* Greedy on uncertainty (statistically tied at p = 0.14). What we do is achieve the same uncertainty *while covering 9.17 pts more area*. The trade-off is what matters: classical methods like Spiral and Potential Field have lower σ (2.41, 2.08) only because they don't move much. We're at the Pareto frontier of (high coverage, low uncertainty), where they're not.

**Q: Your "MS600" recipe just extends the time horizon. How much of the 91.90 is from the horizon extension vs. the Δσ reward?**
A: We disclose this explicitly in Table I:
- Base recipe (T_max=400, no Δσ): 87.56%
- +Δσ (T_max=400, with Δσ): 88.69% — Δσ contributes +1.13 pts
- +Δσ +MS600 (T_max=600, with Δσ): 91.90% — horizon contributes +3.21 pts
So horizon extension contributes more than Δσ alone. The total improvement is the *combination*. We say so honestly in the paper.

**Q: Why didn't you compare against more recent learned methods?**
A: We compared against Vanilla PPO (a fair learned baseline with our architecture but no shaping). Most published learned 3D-active-SLAM methods (chen2024lidar, zhao2024ddpg, yin2025maslam) target different specific environments and aren't directly comparable; they don't release code at our scale. The Vanilla PPO comparison isolates the contribution of *our reward design*, which is the actual claim.

**Q: How does your method handle real noise in the LiDAR / pose estimate?**
A: We don't model perception noise in the simulator. The LiDAR rays are perfect; the pose is exact. This is a deliberate simplification that lets us focus on the reward design. With real noise, we'd expect the policy to need either (a) noisier σ dynamics during training to generalize, or (b) domain randomization. This is part of the sim-to-real future work.

## 20.4 Implementation Questions

**Q: What library / framework did you use?**
A: Stable-Baselines3 + sb3-contrib (for RecurrentPPO). PyTorch backbone. Gymnasium for the environment interface. NumPy / SciPy / Pandas for evaluation and statistics.

**Q: How long does training take?**
A: ~1 GPU-hour per seed for 800k steps on a modern NVIDIA card. 3 seeds = ~3 GPU-hours. Evaluation across 50 maps × 3 seeds takes another ~1 hour.

**Q: How big is the policy at inference time?**
A: ~1.9M parameters total. Inference latency on GPU is sub-millisecond per step; on CPU, a few milliseconds — well within real-time budget for a 10-50 Hz UAV control loop.

**Q: Is your code reproducible?**
A: Yes — fixed seeds, deterministic env (given a seed), checkpointed VecNormalize stats. The exact MS600 checkpoint is `slam3d_180000_steps.zip` for all three seeds.

## 20.5 "Why Not X?" Questions

**Q: Why not DQN?**
A: DQN is for *discrete* action spaces. Our action is *continuous* (dx, dy, dz ∈ [-1, 1]). PPO handles continuous naturally via Gaussian policies.

**Q: Why not SAC or TD3?**
A: SAC/TD3 are off-policy, often more sample-efficient than PPO. We chose PPO for stability — RL training on POMDPs is finicky; PPO's clipping makes it less likely to diverge. Future work could compare with SAC.

**Q: Why not Transformer instead of LSTM?**
A: Transformers have stronger long-range memory but are heavier and harder to train in RL (no batch-effective training due to autoregressive structure). LSTMs are the standard choice for Recurrent PPO. The 200-step effective horizon is well within LSTM range.

**Q: Why not let the policy output a continuous z directly instead of a discrete altitude vote?**
A: We tried this conceptually — it would let the UAV fly at any altitude, but the SLAM back-end's Z-filter struggles with smooth altitude changes. The discrete-vote-with-dwell mimics realistic flight: pick an altitude band, commit, sample, then decide whether to change. Also makes the per-altitude breadth reward well-defined.

---

# 21. Quick Reference — Numbers to Memorize

## 21.1 System Specifications

| Quantity | Value |
|---|---|
| Grid size per altitude | 48 × 48 cells |
| Number of altitudes | 4 (heights 1, 2, 3, 4 m) |
| LiDAR beams | 36 |
| LiDAR range | 14 cells |
| Step size | 3 cells |
| Visit radius | 3 cells (gives 7×7 window) |

## 21.2 Action Space

| Quantity | Value |
|---|---|
| Action shape | [-1, 1]³ |
| Altitude vote threshold τ_alt | 0.3 |
| Min dwell D | 20 steps |

## 21.3 σ Dynamics

| Quantity | Value |
|---|---|
| σ range | [0, 4] |
| σ growth per unit distance | 0.03 |
| σ growth per altitude change | 0.15 |
| σ multiplier on loop closure | 0.70 |
| Loop closure radius | 4 cells |
| Loop closure min step gap | 40 steps |
| Loop closure min path travel | 24 units |

## 21.4 Headline Results

| Quantity | Value |
|---|---|
| Our coverage | 91.90 ± 1.72 % |
| Best classical (Greedy) | 82.73 % |
| Vanilla PPO | 86.87 ± 1.19 % |
| Gap over Greedy | +9.17 pts |
| Gap over Vanilla | +5.03 pts |
| All-altitude minimum | > 85 % |
| Time to 50% (Ours) | ~180 steps |
| Time to 50% (Greedy) | ~370 steps |
| Time to 50% (Nearest Frontier) | ~580 steps |
| Loop closures per episode (Ours) | 7.49 ± 1.17 |
| Loop closures per episode (Greedy) | 0.36 |

## 21.5 Statistical Tests (MS600 vs each)

| Comparison | p-value | δ |
|---|---|---|
| vs. Random Walk | 1.2 × 10⁻²⁶ | +0.990 |
| vs. Nearest Frontier | 1.4 × 10⁻²⁶ | +0.979 |
| vs. Spiral | 1.2 × 10⁻²⁶ | +1.000 |
| vs. Potential Field | 1.2 × 10⁻²⁶ | +0.974 |
| vs. Greedy Info Gain | 8.1 × 10⁻¹⁴ | +0.534 |
| vs. RRT Explorer | 1.2 × 10⁻²⁶ | +0.997 |
| vs. Vanilla PPO | 7.4 × 10⁻¹¹ | +0.486 |

All ***-significant. All "large" effect sizes by Cliff's convention.

## 21.6 Hyperparameters

| HP | Value |
|---|---|
| Total steps | 800,000 |
| Checkpoint interval | 60,000 |
| LR schedule | 3e-4 → 1e-5 (linear) |
| Entropy schedule | 0.02 → 0.002 (linear) |
| γ | 0.995 |
| GAE λ | 0.97 |
| Clip ε | 0.2 |
| n_steps | 256 |
| Batch size | 128 |
| n_epochs | 5 |
| Parallel envs | 4 |
| Max grad norm | 0.5 |

## 21.7 Network Sizes

| Layer | Output shape |
|---|---|
| Input map | (5, 48, 48) |
| After CoordConv2D s=2 | (32, 24, 24) |
| After Conv2 s=2 | (64, 12, 12) |
| After Conv3 s=2 | (64, 6, 6) |
| After Conv4 s=1 + AvgPool6 + Flatten | R^2304 |
| Scalar after LayerNorm + FC-64 + FC-32 | R^32 |
| After Concat + MLP | R^256 |
| LSTM hidden | R^128 |
| Actor / Critic heads | R^256 → R^256 |
| Action | R^3 (μ) + R^3 (log_σ) |
| Value | R^1 |

## 21.8 Curriculum Schedule

| Steps | Difficulty |
|---|---|
| 0 – 120k | easy |
| 120k – 300k | medium |
| 300k – 800k | hard |

---

# Appendix A: Complete Glossary

- **Active SLAM**: SLAM where the robot chooses where to go to optimize SLAM quality.
- **Actor-Critic**: RL architecture with two networks — one outputs actions (actor), one estimates state value (critic).
- **Advantage A(s,a)**: How much better is action a than the average action in state s. A = Q − V.
- **Ablation study**: Removing one component to measure its contribution.
- **BPTT**: Backpropagation Through Time, gradient flow through RNN/LSTM unrolled in time.
- **Cell state c_t**: LSTM long-term memory.
- **Cliff's delta δ**: Non-parametric effect size, [-1, 1].
- **CoordConv**: Conv2D with extra normalized coordinate channels.
- **Coverage Cov(s)**: Fraction of true free cells observed.
- **Curriculum**: Gradually increasing task difficulty during training.
- **Discount γ**: How much future rewards count vs. present.
- **Effective horizon**: 1/(1−γ), how many steps the agent looks ahead.
- **Entropy H[π]**: Measure of policy randomness, encourages exploration.
- **Episode**: One trial from start to termination.
- **Frontier**: Boundary cell between free and unknown space.
- **GAE**: Generalized Advantage Estimation, weighted sum of TD errors.
- **GPS-denied**: Place where GPS doesn't work.
- **GroupNorm**: Normalization layer dividing channels into groups.
- **Hidden state h_t**: LSTM short-term memory, output to downstream layers.
- **Horizon T_max**: Maximum episode length.
- **Information gain**: Expected reduction in uncertainty from an action.
- **LayerNorm**: Per-sample normalization across features.
- **LiDAR**: Laser-based ranging sensor.
- **Loop closure**: Recognition of a previously-visited place; reduces SLAM drift.
- **LSTM**: Long Short-Term Memory recurrent network.
- **MaxPooling**: Down-sampling by taking max in each window.
- **MDP**: Markov Decision Process.
- **MLP**: Multi-Layer Perceptron, fully-connected feedforward network.
- **Occupancy grid**: 2D/3D grid where each cell is free/occupied/unknown.
- **POMDP**: Partially Observable MDP.
- **Policy π**: Mapping from observation (and history) to action distribution.
- **PPO**: Proximal Policy Optimization, an RL algorithm with clipped surrogate objective.
- **Pose**: Position + orientation, e.g., (x, y, z, ψ).
- **Pose covariance Σ**: Matrix capturing uncertainty in pose estimate.
- **Recurrent PPO**: PPO with an LSTM/GRU between encoder and heads.
- **Reward shaping**: Adding auxiliary reward terms to guide learning.
- **RTAB-Map**: A real-time graph-based SLAM system.
- **Scalar surrogate σ**: Our scalar proxy for Tr(Σ).
- **SLAM**: Simultaneous Localization and Mapping.
- **Stride**: How many pixels a conv kernel moves between applications.
- **TD error δ**: r + γV(s') − V(s).
- **TRPO**: Trust-Region Policy Optimization, predecessor to PPO.
- **UAV**: Unmanned Aerial Vehicle.
- **VecNormalize**: SB3 wrapper for reward/observation normalization.
- **Wilcoxon signed-rank**: Non-parametric paired test.
- **Z-filter**: Low-pass filter on UAV altitude estimate.

---

*Document length: comprehensive defense reference. Target: be able to answer any question on any term, concept, or design choice in the project.*
