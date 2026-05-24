# Project Explanation: Residual Coordination for Maritime Multi-UAV Search and Tracking Under Sparse Communication

**Author:** Satish Kumar Mishra, Netaji Subhas University of Technology, Delhi  
**Mentor:** Dr. Rashmi Gupta  
**Code:** https://github.com/satishmishra135799-maker/Residual_Swarm  
**Paper:** `paper/report_final.tex`

---

## STEP 1: The Problem — Why Does This Project Exist?

### 1.1 The Real-World Scenario
Imagine a ship sinks in the middle of the ocean. Survivors are floating in the water. Every minute counts. You need to find them fast.

You send drones. But here's the brutal reality:
- The ocean is huge. One drone can't cover it.
- Survivors drift — ocean currents and wind move them. They're not sitting still.
- Each drone can only see a small area around itself (limited sensor range).
- Drones can't always talk to each other — they're spread out, communication breaks.
- No drone knows the full picture — each one only knows what it personally saw.

This is the problem this project solves.

---

### 1.2 Why Is This Hard? (The 3 Core Tensions)

**Tension 1: Search vs Track**
When a drone finds a victim, should it stay and keep watching (track), or keep moving to find more victims (search)? If it leaves, the victim drifts away and gets lost. If it stays, other victims are never found.

**Tension 2: Explore vs Exploit**
Should drones go to areas they haven't checked yet (explore), or go back to areas where they think victims probably are (exploit their belief)?

**Tension 3: Individual vs Team**
Each drone acts alone with local information. But the team needs coordinated behavior. How do you get coordination without constant communication?

---

### 1.3 The Formal Name for This Problem
In AI, this is called a **Dec-POMDP**:
- **Dec** = Decentralized (no single brain controlling all drones)
- **POMDP** = Partially Observable Markov Decision Process

| Word | Meaning in this project |
|---|---|
| Decentralized | Each drone decides for itself |
| Partially Observable | Each drone only sees nearby area |
| Markov | Future depends only on current state, not history |
| Decision Process | Drones must choose actions at each step |

---

### 1.4 What the Project Actually Builds
The project builds a simulator of this maritime rescue scenario and then tests different strategies (called policies) for how drones should move. It answers one central question:

> Can a drone that follows smart handcrafted rules be made even smarter by adding a small learned correction on top — without throwing away the rules?

**The answer the paper finds: Yes, and it's especially useful in harder conditions.**

---

### 1.5 The Setup in Numbers

| Parameter | Default | Medium | Hard |
|---|---|---|---|
| Area size | 1000×1000 m | same | same |
| Number of drones | 4 | 5 | 6 |
| Number of victims | 2 | 3 | 4 |
| Sensing radius | 140 m | 120 m | 95 m |
| Communication radius | 260 m | 220 m | 180 m |
| Victim drift speed | (2.0, 1.0) m/step | (3.0, 1.7) | (4.0, 2.5) |
| Simulation steps | 80 | 95 | 110 |

As you go from default → medium → hard: more victims, smaller sensing radius, faster drift, weaker communication. The problem gets genuinely harder in every dimension simultaneously.

---

### 1.6 What Success Looks Like
The project measures 6 metrics:
1. **Detection Rate** — what fraction of victims were found at least once
2. **Tracking Ratio** — once found, how consistently was the victim kept in sight
3. **Coverage Mean** — how much of the search area was visited
4. **First Detection Step** — how quickly was the first victim found (lower = better)
5. **Track Loss Count** — how many times was a victim found then lost
6. **Reacquisition Delay** — after losing a victim, how long to find it again

The project's main claim: residual learning improves tracking and first detection most consistently, across all three difficulty levels.

---

## STEP 2: The Simulator — How the World is Built in Code

### 2.1 The Big Picture
The simulator is one file: `src/swarm_sim/simulator.py` — the `MaritimeSwarmSimulator` class. It is the entire world. Everything that happens — drones moving, victims drifting, detections, communication, belief updates — all happens inside this one class.

Think of it as a game engine where:
- The world ticks forward one step at a time
- Each tick = one call to `step()`
- The simulation runs for a fixed number of steps (80/95/110 depending on scenario)

---

### 2.2 What Exists in the World
At any moment, the simulator holds:

```
uav_positions    → shape (N, 2) — where each drone is
uav_velocities   → shape (N, 2) — how fast each drone is moving
victim_positions → shape (M, 2) — where each victim is
victim_drifts    → shape (M, 2) — how fast each victim drifts
beliefs          → list of N grids — each drone's probability map
coverage         → 20×20 grid — how recently each area was visited
```

N = number of drones, M = number of victims. Everything is 2D (x, y).

---

### 2.3 One Simulation Step — The Exact Sequence
Every call to `step()` does these things in this exact order:

1. **SENSE** → each drone checks if any victim is within sensing_radius
2. **THINK** → compute belief peaks, coverage gaps
3. **DECIDE** → run the policy to get actions (or use external actions)
4. **MOVE** → update drone positions using those actions
5. **DRIFT** → move victims according to their drift vectors + noise
6. **UPDATE MAP** → update the coverage grid (mark visited cells)
7. **SENSE AGAIN** → re-detect after movement
8. **UPDATE BELIEF** → Bayesian update: predict → observe → fuse messages
9. **COMMUNICATE** → event-triggered: send belief to neighbors if needed
10. **RECORD** → log everything into a StepRecord

The order matters — especially that sensing happens before and after movement.

---

### 2.4 How Drones Are Placed at Start
From `reset()`:
```python
lane_width = width / num_uavs
uav_positions = [(lane_width * (i + 0.5), height * 0.1) for i in range(num_uavs)]
```
Drones start evenly spaced along the bottom of the search area, each in its own lane. This gives them a structured starting spread so they don't all cluster together.

---

### 2.5 How Victims Move (Drift)
Each victim has its own drift vector. They don't all drift the same way:
```python
offsets = linspace(-1.0, 1.0, num_victims)
drift[i] = [drift_x + offset * spread, drift_y + 0.6 * offset * spread]
```
So if drift_x = 2.0 and spread = 0.7:
- Victim 0 drifts at (1.3, ...)
- Victim 1 drifts at (2.7, ...)

Each step, random noise is also added (victim_noise_std = 3.0). This makes victims unpredictable — you can't just extrapolate perfectly.

---

### 2.6 How Detection Works
A drone detects a victim if the Euclidean distance is within sensing_radius:
```python
distances = ||uav_positions - victim_positions||  # shape (N, M)
visible = distances <= sensing_radius
detections = any(visible, axis=1)  # shape (N,) — did drone i see anything?
```
Simple circle-based sensing. No false positives, no false negatives in the current model (acknowledged as a limitation in the paper).

---

### 2.7 How Communication Works
Two drones can communicate only if they are within communication_radius of each other. But communication is not always used even when a link exists. A drone only sends its belief if one of these is true:
- It just detected a victim
- Its belief confidence is above a threshold (0.14)
- Its confidence changed significantly (by 0.025)
- It hasn't sent a message in 6 steps (staleness timer)

This is **event-triggered communication** — sparse by design to save bandwidth.

---

### 2.8 The Coverage Grid
The coverage grid is a 20×20 array representing the 1000×1000 area. Each cell is 50×50 m.
- Starts at all zeros (nothing visited)
- Every step: all cells decay by factor 0.96 (coverage × 0.96)
- Cells within sensing_radius of any drone are set to 1.0

The decay means old visits fade. A cell visited 20 steps ago has value 0.96^20 ≈ 0.44. This forces drones to revisit stale areas rather than assuming they're still covered.

The **coverage gap target** = the cell with the lowest value in a drone's assigned lane. This is what frontier-based policies chase.

---

### 2.9 Tracking vs Detection — The Difference

| | Detection | Tracking |
|---|---|---|
| What it means | Victim entered sensing radius | Victim within tighter track_distance (0.57 × r_s) |
| Sensing radius (default) | 140 m | 80 m |
| Metric | detection_rate | tracking_ratio |
| Harder to maintain? | No | Yes — victim drifts out of tight range |

A drone can detect a victim without tracking it (victim is between 80–140 m away). Tracking requires staying close. This is why tracking is the harder metric to optimize.

---

### 2.10 What Gets Recorded Each Step
Every step produces a StepRecord containing:
- All drone and victim positions
- Which drones detected which victims
- Which drones are assigned to which victims
- Communication links and who transmitted
- Belief peak confidences
- Event markers: first_detection, track_loss, handoff_success, etc.

These event markers are how the reward function and metrics are computed.


---

## STEP 3: Belief Maps — How Each Drone Thinks About Where Victims Are

### 3.1 The Core Problem
Each drone can only see a small circle around itself (radius 140 m in a 1000×1000 m world). Most of the time, it sees nothing. But it still needs to make smart decisions about where to go.

The solution: each drone maintains a **belief map** — a probability distribution over the entire search area representing "where do I think the victim currently is?"

---

### 3.2 What a Belief Map Actually Is
It's a 20×20 grid of numbers that sum to 1.0.

Grid cell (row, col) = probability that a victim is in that 50×50 m area.

At the start, every cell has equal probability = 1/400 = 0.0025. The drone knows nothing, so it assumes the victim could be anywhere equally. As the simulation runs, this grid gets sharper — high probability concentrates where the drone thinks the victim is.

---

### 3.3 The Three-Stage Belief Update
Every step, each drone's belief goes through exactly 3 stages:
- **Stage 1: PREDICT** → account for victim drift
- **Stage 2: OBSERVE** → update based on what the drone just saw (or didn't see)
- **Stage 3: FUSE** → incorporate messages from neighbors

---

### 3.4 Stage 1 — Predict (Drift Propagation)
The drone knows victims drift. So before updating with new observations, it shifts the belief map in the direction of drift.

From `belief.py`:
```python
shift_x = round(drift_x / cell_width)   # how many cells to shift
shift_y = round(drift_y / cell_height)
shifted = roll(belief, shift=(shift_y, shift_x))  # numpy roll = circular shift
```
Then it blurs slightly to account for uncertainty:
```python
blurred = (shifted + neighbors) / 5.0
result = (1 - blend) * shifted + blend * blurred
```
belief_blend = 0.35 — so 65% stays sharp, 35% gets blurred. This prevents the belief from becoming overconfident about exact position after drift.

**Intuition:** If you last saw a victim at position X, and you know it drifts east at 2 m/step, your best guess is now X + 2 m east — but with some spread because drift has noise.

---

### 3.5 Stage 2 — Observe (Bayesian Update)

**Case A: Drone detected a victim**
The belief is replaced with a Gaussian centered on the detected victim's position:
```python
gaussian[row, col] = exp(-((row - target_row)² + (col - target_col)²) / 3.0)
result = 0.25 * old_belief + 0.75 * gaussian
```
The belief snaps strongly toward where the victim actually is.

**Case B: Drone saw nothing**
The cells the drone could see (within sensing_radius) get suppressed:
```python
belief[visible_cells] *= negative_observation_decay  # multiply by 0.35
belief = normalize(belief)
```
Multiplying by 0.35 means "I looked here and saw nothing, so it's much less likely the victim is here." The remaining probability mass redistributes to unseen areas.

This is Bayesian reasoning in action:
- Positive evidence → concentrate belief at detection location
- Negative evidence → suppress belief in searched area

---

### 3.6 Stage 3 — Fuse (Message Integration)
If a neighbor drone sent its belief map, the drone combines them:
```python
fused = local_belief * incoming_belief  # element-wise multiply
fused = normalize(fused)
```
Multiplying two probability distributions = keeping only what both agree on. If drone A thinks the victim is in the top-left and drone B agrees, the fused belief is very confident about top-left. If they disagree, the result is more spread out.

This is called **belief consensus** — a lightweight form of distributed Bayesian fusion.

---

### 3.7 What "Belief Peak" and "Belief Confidence" Mean
After all three stages, the drone extracts two numbers:
```python
row, col = argmax(belief)
peak_xy = [(col + 0.5) * cell_width, (row + 0.5) * cell_height]
peak_confidence = belief[row, col]
```

These two numbers are what the policies actually use to make decisions:
- peak_confidence > 0.08 → "I'm confident enough, go chase the belief peak"
- peak_confidence < 0.08 → "I'm not sure, keep exploring"

**The threshold 0.08 is the project's central decision boundary** — used by the hybrid policy AND the gating mechanism.

---

### 3.8 The Key Numbers to Remember

| Parameter | Value | Meaning |
|---|---|---|
| Grid size | 20×20 | 400 cells over 1000×1000 m area |
| Cell size | 50×50 m | resolution of belief |
| belief_blend | 0.35 | how much drift blurring |
| negative_observation_decay | 0.35 | how strongly negative evidence suppresses |
| message_confidence_threshold | 0.14 | minimum confidence to trigger a message |
| decision threshold | 0.08 | when to switch from explore to track |

---

## STEP 4: Handcrafted Policies — The 4 Baseline Strategies

### 4.1 What is a Policy?
A policy is simply a function: **observation → action**

Given what a drone currently knows (its position, belief peak, coverage gap, etc.), the policy outputs a velocity vector — which direction and how fast to move. All handcrafted policies output a 2D vector clipped to max_speed = 28 m/step.

---

### 4.2 The Assignment System (Shared by ALL Policies)
Before any policy runs, the simulator runs an assignment engine that decides: "Which drone should chase which victim right now?"

The logic:
1. Drones that currently see a victim → assigned to that victim
2. Remaining drones → assigned to highest-priority untracked victims
3. Leftover drones → sent to explore coverage gaps

Victim priority is scored by:
- Not yet discovered → +2.0 bonus
- Hasn't been seen in a long time → up to +5.0 staleness bonus
- Currently untracked → +1.1 bonus
- Fast drifting → +0.2 bonus

Once assigned, the drone gets a `target_hint` — the estimated victim position. Every policy checks `target_hint` first. If it's not None, the drone goes straight there regardless of policy type.

**This means the policies only differ in what they do when no victim is assigned — i.e., during pure exploration.**

---

### 4.3 Policy 1: sweep_only
The simplest policy. No belief, no coverage map.
```python
lane_x = (uav_index + 0.5) * (width / num_uavs)
target_y = height * 0.9 if moving_up else height * 0.1
→ move toward (lane_x, target_y)
```
Each drone is locked to its vertical lane and sweeps up and down like a lawnmower.

**Strengths:** Very fast first detection in easy settings — systematic coverage guarantees early contact.

**Weaknesses:** Completely ignores where victims actually are. Terrible tracking — drone keeps sweeping even after finding a victim. Degrades badly in hard scenarios.

---

### 4.4 Policy 2: frontier_cover
Exploration-first. Uses coverage gap, ignores belief.
```python
lane_x = (uav_index + 0.5) * (width / num_uavs)
lane_anchor = [lane_x, coverage_target[1]]
target = 0.7 * coverage_target + 0.3 * lane_anchor
→ move toward target
```
The `coverage_target` is the least recently visited cell in the drone's lane.

**Strengths:** Best coverage of all policies. Strong tracking in hard scenarios. Low variance — consistent behavior across seeds.

**Weaknesses:** Slower first detection — it doesn't rush toward likely victim locations. Ignores belief entirely.

---

### 4.5 Policy 3: belief_sparse_comm
Belief-heavy. Blends sweep + belief + coverage.
```python
if peak_confidence > 0.06:
    target = 0.60 * belief_peak + 0.25 * coverage_target + 0.15 * sweep_target
else:
    target = 0.55 * coverage_target + 0.45 * sweep_target
```
Uses a lower confidence threshold (0.06) to start chasing the belief peak. Weights belief very heavily (60%) once confident.

**Strengths:** Fast detection in easy settings — quickly exploits belief signals.

**Weaknesses:** Over-relies on belief — if belief is wrong (victim drifted), drone chases the wrong location. Worst performance in hard scenarios where drift makes belief unreliable.

---

### 4.6 Policy 4: hybrid_frontier_belief (The Expert Base)
The main policy. Balances frontier exploration and belief pursuit.
```python
if peak_confidence > 0.08:
    target = 0.55 * belief_peak + 0.30 * coverage_target + 0.15 * lane_anchor
else:
    target = 0.75 * coverage_target + 0.25 * lane_anchor
```

**Two modes, one threshold:**

*Exploration mode (confidence ≤ 0.08):*
- 75% weight on coverage gap → go explore stale areas
- 25% weight on lane anchor → stay loosely in your lane

*Tracking mode (confidence > 0.08):*
- 55% weight on belief peak → chase where victim probably is
- 30% weight on coverage gap → don't completely abandon exploration
- 15% weight on lane anchor → maintain loose spatial structure

**Why threshold 0.08?** It's calibrated to be higher than noise-level belief values but lower than the communication threshold (0.14). It means "I have real evidence, not just random fluctuation."

**Why is this chosen as the residual learning base?** Not because it dominates every metric — frontier_cover actually beats it on tracking and coverage in hard scenarios. It's chosen because it exposes communication-aware, belief-driven behavior that can be meaningfully improved by learning.

---

### 4.7 Side-by-Side Comparison

| | sweep_only | frontier_cover | belief_sparse_comm | hybrid (expert) |
|---|---|---|---|---|
| Uses belief? | No | No | Yes (heavy) | Yes (moderate) |
| Uses coverage? | No | Yes (primary) | Yes (secondary) | Yes (secondary) |
| Confidence threshold | — | — | 0.06 | 0.08 |
| Best at | Early detection | Coverage + tracking | Fast detection | Balanced |
| Worst at | Hard scenarios | First detection speed | Hard scenarios | Nothing specific |

---

### 4.8 The Action Formula (All Policies)
Every policy ultimately computes a target point, then:
```python
delta = target - current_position
norm = ||delta||
action = (delta / norm) * max_speed  # unit vector × speed
```
If the drone is already very close to the target (norm < 20), it adds small random jitter to avoid getting stuck:
```python
jitter = rng.normal(0.0, 8.0, size=2)
target = clip(target + jitter, [0,0], [width, height])
```
The output is always a velocity vector of magnitude max_speed = 28. Drones always move at full speed.


---

## STEP 5: The MLP Policy — The Neural Network

### 5.1 Why a Neural Network at All?
The handcrafted policies work, but they have fixed rules. They can't adapt. If the environment changes — stronger drift, weaker communication, more victims — the weights (0.55, 0.30, 0.15) stay the same. They were tuned by hand and stay frozen.

A neural network can learn better weights from experience. But there's a problem: learning the full movement action from scratch failed in this project. The paper tried it and it didn't work well. So instead, the network only learns a small correction on top of the expert — that's the residual idea (Step 6). But first, you need to understand the network itself.

---

### 5.2 The Architecture — Brutally Simple
From `mlp_policy.py`:
```
Input layer:  25 numbers (observation vector)
Hidden layer: 16 neurons (tanh activation)
Output layer: 2 numbers (x correction, y correction)
```
That's it. One hidden layer. 25 → 16 → 2.
```python
hidden = tanh(W1 @ observation + b1)  # shape: (16,)
output = tanh(W2 @ hidden + b2)       # shape: (2,)
```
The tanh activation squashes everything to the range (-1, 1). The output is always between -1 and 1 — it represents a normalized correction, not a raw velocity.

**Total parameters:**
- W1: 16 × 25 = 400
- b1: 16
- W2: 2 × 16 = 32
- b2: 2
- **Total = 450 numbers**

450 numbers to represent the entire learned policy. Extremely lightweight — this runs in pure NumPy with no GPU needed.

---

### 5.3 The Observation Vector — What the Network Sees
Each drone gets a 25-dimensional vector. Every number is normalized to roughly (-1, 1) range.

| Index | Value | What it represents |
|---|---|---|
| 0 | pos_x / width | Drone's x position (normalized) |
| 1 | pos_y / height | Drone's y position (normalized) |
| 2 | vel_x / max_speed | Drone's x velocity (normalized) |
| 3 | vel_y / max_speed | Drone's y velocity (normalized) |
| 4 | peak_x / width | Belief peak x position |
| 5 | peak_y / height | Belief peak y position |
| 6 | peak_confidence | How confident the belief is (0–1) |
| 7 | gap_x / width | Coverage gap target x |
| 8 | gap_y / height | Coverage gap target y |
| 9 | neighbors / (N-1) | Fraction of drones in communication range |
| 10 | detection_flag | 1 if currently detecting a victim, else 0 |
| 11 | peak_delta_x | (peak_x - pos_x) / width — relative direction to belief peak |
| 12 | peak_delta_y | (peak_y - pos_y) / height |
| 13 | gap_delta_x | (gap_x - pos_x) / width — relative direction to coverage gap |
| 14 | gap_delta_y | (gap_y - pos_y) / height |
| 15 | global_peak_delta_x | Direction to swarm-wide average belief peak |
| 16 | global_peak_delta_y | Direction to swarm-wide average belief peak |
| 17 | global_peak_conf | Confidence of swarm-wide average belief |
| 18 | local_coverage_value | How recently this drone's current cell was visited |
| 19 | last_action_x | What action the drone took last step |
| 20 | last_action_y | What action the drone took last step |
| 21 | step_fraction | How far through the episode (0 at start, 1 at end) |
| 22 | lane_fraction | Which lane this drone belongs to (0 to 1) |
| 23 | assigned_victim_norm | Which victim is currently assigned (normalized) |
| 24 | handoff_target_flag | 1 if this drone currently owns a victim |

**Key design choices:**
- Inputs 11–16 are relative (delta) not absolute. The network learns "go toward belief peak" not "go to coordinate (400, 300)". This makes it generalize across positions.
- Input 17 (global_peak_conf) gives each drone a team-level signal.
- Input 21 (step_fraction) lets the network behave differently early vs late in the episode.
- Input 24 (handoff_target_flag) tells the drone if it's responsible for a victim — critical for tracking behavior.

---

### 5.4 How the Network is Stored
The network weights are stored as a single flat array called theta:
```
theta = [W1 flattened (400), b1 (16), W2 flattened (32), b2 (2)]
# total length = 450
```
The `unpack(theta)` function reshapes this back into W1, b1, W2, b2. This flat representation is used because the training algorithm (evolution strategy) treats the entire network as one vector to optimize — it doesn't do backpropagation.

---

### 5.5 How the Network is Trained — Evolution Strategy (Updated: 50 Iterations)
No gradient descent. No backpropagation. No PyTorch. The training uses a **population-based evolution strategy**:

1. Start with theta = zeros (450 numbers)
2. Generate 12 candidate thetas by adding random noise to current best
3. Run each candidate through the full simulation (3 scenarios × 3 seeds)
4. Score each candidate using the shaped reward
5. Keep the top 4 (elites)
6. New mean = average of top 4
7. New std = std of top 4 (but minimum 0.03 to keep exploring)
8. **Repeat for 50 iterations** (updated from original 10)

**Why not backpropagation?** The entire project runs on NumPy only. Evolution strategies work fine with NumPy — you just need to evaluate the network many times, which is cheap because the simulator is fast.

---

### 5.6 The Reward/Score Function During Training
Each candidate theta is scored by running it through the simulator and computing:
```
score = total_reward_across_steps
      + 70.0 × tracking_ratio
      + 45.0 × detection_rate
      + 20.0 × coverage_mean
      + max(0, 25 - first_detection_step)   # bonus for early detection
      - 1.5 × track_loss_count              # penalty for losing victims
```
The weights (70, 45, 20) reveal the project's priorities:
- Tracking is most important (weight 70)
- Detection is second (weight 45)
- Coverage is third (weight 20)

Scenarios are also weighted during training:
- default: ×0.8
- medium: ×1.0
- hard: ×1.35

The hard scenario gets the most weight — the system is explicitly trained to be robust in difficult conditions.

---

### 5.7 Why Direct Learning Failed
The paper tried three approaches before residual learning:

| Approach | What it does | Why it failed |
|---|---|---|
| Direct MLP | Network outputs full action | Unstable — network has to learn everything from scratch |
| Behavior Cloning | Imitate expert demonstrations | Couldn't recover expert quality — distribution shift problem |
| DAgger | Iterative imitation with corrections | Still weaker than handcrafted expert in this environment |

All three are implemented in the repo. They exist as evidence that the simpler approaches were tried and failed, motivating the residual approach.

The fundamental problem: the search space is too large for a small network to learn from scratch in limited iterations with 12 candidates. The residual approach shrinks the learning problem — instead of learning "how to fly a drone in a maritime rescue," it only learns "how to slightly adjust an already-good policy."

---

## STEP 6: Residual Learning — The Core Contribution

### 6.1 The Central Idea in One Sentence
Instead of replacing the expert, add a small learned correction on top of it.

---

### 6.2 Why "Residual"?
The word comes from mathematics. A residual is the difference between what you have and what you want:
```
residual = desired_output - current_output
```
In this project:
- expert gives: a_base (good but not perfect)
- network learns: a_res (the gap between good and better)
- final action: a_base + a_res

The network doesn't need to learn how to fly a drone. It only needs to learn what the expert is getting wrong — a much smaller, easier problem.

This is the same idea used in ResNet (deep learning), boosting algorithms, and PID controllers with learned feedforward terms. It's a general principle: build on structure rather than replace it.

---

### 6.3 The Three Variants (All Novel Contributions)

---

### 6.4 Variant 1 — Fixed Residual (The Paper's Main Method)
From `env_interface.py`:
```python
expert = builtin_actions / max_speed          # normalize to (-1, 1)
final = clip(expert + alpha × network_output, -1, 1)
```
Where alpha = residual_scale = **0.25** (the frozen paper setting).

**Visualized:**
```
Expert action:      →→→  (pointing northeast, magnitude 0.8)
Network correction: ↑    (small upward nudge, magnitude 0.3)
alpha × correction: ↑    (scaled down to 0.075)
Final action:       →→↑  (mostly expert, slightly adjusted)
clip to (-1,1):     →→↑  (stays within bounds)
```

**The alpha ablation — four values tested (updated results with 50 iterations):**

| alpha | Tracking (Hard) | First Det. (Hard) | Coverage (Hard) |
|---|---|---|---|
| 0.00 | 0.905 ± 0.003 | 3.37 ± 0.72 | 0.295 ± 0.034 |
| 0.10 | 0.909 ± 0.003 | 2.67 ± 0.76 | 0.270 ± 0.020 |
| 0.25 | 0.915 ± 0.003 | 2.30 ± 0.65 | 0.266 ± 0.006 |
| 0.40 | 0.918 ± 0.003 | 2.07 ± 0.45 | 0.274 ± 0.006 |

The fact that alpha=0.0 exactly reproduces the expert is the validation check — it proves the residual wrapper is implemented correctly.

**Why 0.25 is the default:** Small enough that the expert's search structure is preserved. Large enough that the network can meaningfully adjust tracking behavior. The marginal gain from alpha=0.4 is not statistically distinguishable.

---

### 6.5 Variant 2 — Binary-Gated Residual (Novel)
From `env_interface.py`:
```python
expert = builtin_actions / max_speed
residual = network_output
# compute gate per drone
confidences = [obs.belief_confidence for obs in observations]
gates = (confidences >= 0.08).astype(float)  # 1.0 or 0.0
final = clip(expert + alpha × gates[:, None] × residual, -1, 1)
```

The gate is **per drone, per step**. Each drone independently decides whether to apply the correction.

**What this means in practice:**
```
Drone 0: confidence = 0.12 → gate = 1.0 → correction applied
Drone 1: confidence = 0.04 → gate = 0.0 → pure expert
Drone 2: confidence = 0.09 → gate = 1.0 → correction applied
Drone 3: confidence = 0.02 → gate = 0.0 → pure expert
```

At any given step, some drones are in "learning mode" and others are in "pure expert mode" — depending on what each drone individually believes.

**The intuition:**
- When confidence is low → drone is exploring blindly → expert's frontier behavior is best → don't interfere
- When confidence is high → drone has found something → this is where coordination matters → let the network improve it

The threshold 0.08 is deliberately the same as the hybrid policy's own decision threshold. The gate activates exactly when the expert switches from exploration mode to tracking mode.

**No existing swarm residual policy applies state-dependent gating at the individual agent level** — this is the novel contribution.

---

### 6.6 Variant 3 — Soft-Gated Residual (Novel Extension)
From `env_interface.py`:
```python
k = 20.0
gates = 1.0 / (1.0 + exp(-k × (confidences - 0.08)))
```
This is a sigmoid function centered at 0.08 with steepness k=20.

**What the gate value looks like:**

| Confidence | Gate value |
|---|---|
| 0.02 | ≈ 0.000 (essentially off) |
| 0.06 | ≈ 0.018 (nearly off) |
| 0.08 | = 0.500 (exactly half) |
| 0.10 | ≈ 0.982 (nearly on) |
| 0.14 | ≈ 1.000 (fully on) |

Instead of snapping from 0 to 1, the gate smoothly interpolates. At exactly the threshold, the correction is applied at half strength.

**Why smooth gating?** The hard gate can cause sudden behavioral changes when confidence crosses 0.08. If a drone's confidence oscillates around the threshold (0.079 → 0.081 → 0.079), the correction switches on and off every step, causing jitter. The soft gate eliminates this by making the transition gradual.

---

### 6.7 The Three Variants Side by Side
```
Fixed Residual:
  always: final = expert + 0.25 × correction

Binary-Gated Residual:
  if conf ≥ 0.08: final = expert + 0.25 × correction
  if conf < 0.08: final = expert  (pure expert)

Soft-Gated Residual:
  always: final = expert + 0.25 × sigmoid(20×(conf-0.08)) × correction
  (correction weight smoothly varies from ~0 to ~0.25)
```

---

### 6.8 Why This Works — The Systems Explanation
The expert is good at structure. The network is good at adaptation.

**The expert knows:**
- How to sweep lanes
- When to switch from explore to track
- How to assign drones to victims
- When to send messages

**The network learns:**
- How to adjust tracking direction when drift is strong
- How to respond faster when a victim is first detected
- How to coordinate handoffs more smoothly
- When to slightly prioritize coverage vs tracking based on episode progress

Neither alone is optimal. Together, they cover each other's weaknesses.

---

### 6.9 The Training Loop for Residual Learning
The residual network is trained the same way as the direct MLP (evolution strategy), but the evaluation is different:
```python
# During training rollout:
expert = simulator.current_builtin_actions() / max_speed
correction = network_forward(observation)
final_action = clip(expert + 0.25 × correction, -1, 1)
simulator.step(external_actions = final_action)
```
The network never sees the simulator without the expert. It always trains on top of the expert. This means:
- The network can't unlearn the expert's good behavior
- It can only learn to improve from the expert's baseline
- If the network learns nothing useful, alpha=0 reproduces the expert exactly (safe fallback)

**This is the key safety property of residual learning: it can only help, never catastrophically hurt (as long as alpha is small enough).**

---

### 6.10 The Complete Action Pipeline (Everything Together)
Here is the full flow from observation to action in the residual system:

1. Simulator state → build 25-dim observation vector per drone
2. Observation → MLP forward pass → 2-dim correction ∈ (-1,1)
3. Simulator state → hybrid expert → 2-dim base action
4. Normalize base action to (-1,1) range
5. Compute gate (1.0, 0/1, or sigmoid) based on belief confidence
6. final = clip(base + alpha × gate × correction, -1, 1)
7. Denormalize: final × max_speed = actual velocity
8. Apply to simulator

Every drone runs this pipeline independently, every step, in parallel.


---

## STEP 7: Evaluation & Results — The Full Experimental Story (Updated)

### 7.1 The Evaluation Philosophy
The paper makes one deliberate design choice that separates it from weak research: **matched-seed evaluation**.

Every method is tested on the exact same **30 random seeds** (updated from original 10):
```
seeds = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39,
         43, 47, 51, 55, 59, 63, 67, 71, 75, 79,
         83, 87, 91, 95, 99, 103, 107, 111, 115, 119]
```
Same seeds = same victim starting positions, same drift noise, same initial conditions. This means differences in results are due to the policy, not luck. Statistical significance is confirmed with **paired t-tests**.

---

### 7.2 The Three Scenarios

| Parameter | Default | Medium | Hard |
|---|---|---|---|
| Drones | 4 | 5 | 6 |
| Victims | 2 | 3 | 4 |
| Sensing radius | 140 m | 120 m | 95 m |
| Comm radius | 260 m | 220 m | 180 m |
| Drift speed | (2.0, 1.0) | (3.0, 1.7) | (4.0, 2.5) |
| Steps | 80 | 95 | 110 |

---

### 7.3 Table 1: Five-Method Comparison (Tracking Ratio, 30 Seeds)

| Scenario | Hybrid | Pure MLP | Fixed | Binary Gate | Soft Gate |
|---|---|---|---|---|---|
| Default | 0.869 | 0.251 | **0.882** | 0.879 | 0.879 |
| Medium | 0.879 | 0.148 | 0.876 | **0.887** | 0.890 |
| Hard | 0.905 | 0.393 | **0.915** | 0.907 | 0.910 |
| First Det. (Hard) | 3.37 | 4.90 | **2.30** | 3.37 | 3.13 |
| Coverage (Hard) | 0.296 | — | 0.266 | 0.268 | **0.274** |

**The pure MLP achieves tracking ratios of 0.25, 0.15, and 0.39** — far below the hybrid controller (0.87, 0.88, 0.91). This confirms that end-to-end learning without a structured base controller is substantially weaker, even with 50 ES iterations. The structured base controller provides essential inductive bias.

---

### 7.4 Table 2: Fixed Residual vs Hybrid (Mean ± Std, 30 Seeds)

| Scenario | Track. ↑ | Miss ↓ | First Det. ↓ | Cov. |
|---|---|---|---|---|
| Default (Hybrid) | 0.869 ± 0.008 | 0.132 | 2.83 ± 0.65 | 0.466 |
| Default (Fixed) | 0.882 ± 0.009 | 0.118 | 2.53 ± 0.76 | 0.463 |
| Medium (Hybrid) | 0.879 ± 0.046 | 0.121 | 1.83 ± 0.79 | 0.462 |
| Medium (Fixed) | 0.876 ± 0.040 | 0.124 | 1.60 ± 0.74 | 0.460 |
| Hard (Hybrid) | 0.905 ± 0.003 | 0.095 | 3.37 ± 0.72 | 0.296 |
| Hard (Fixed) | 0.915 ± 0.003 | 0.085 | 2.30 ± 0.63 | 0.266 |

All 30 runs achieve discovery ratio 1.0 for both methods. Paired t-tests confirm significance: tracking improvement is p < 0.001 in default and hard scenarios, p = 0.044 in medium.

**Key result:** Hard scenario first detection improved from 3.37 → 2.30 steps (p < 0.001). At 4 m/step drift, this corresponds to ~4.3 m less positional uncertainty at the moment of detection.

---

### 7.5 Table 3: Three-Way Comparison — Hybrid, Fixed, Binary Gate (30 Seeds)

| Scenario | Track. ↑ | Miss ↓ | First Det. ↓ | Cov. |
|---|---|---|---|---|
| Default (H) | 0.869 ± 0.008 | 0.132 | 2.83 | 0.466 |
| Default (F) | 0.882 ± 0.009 | 0.118 | 2.53 | 0.463 |
| Default (G) | 0.879 ± 0.008 | 0.121 | 2.83 | 0.468 |
| Medium (H) | 0.879 ± 0.046 | 0.121 | 1.83 | 0.462 |
| Medium (F) | 0.876 ± 0.040 | 0.124 | 1.60 | 0.460 |
| Medium (G) | 0.887 ± 0.040 | 0.113 | 1.83 | 0.451 |
| Hard (H) | 0.905 ± 0.003 | 0.095 | 3.37 | 0.296 |
| Hard (F) | 0.915 ± 0.003 | 0.085 | 2.30 | 0.266 |
| Hard (G) | 0.907 ± 0.003 | 0.093 | 3.37 | 0.268 |

The gated residual improves tracking over hybrid in all scenarios (p < 0.001 default and hard, p = 0.109 medium). In default and medium, gated recovers coverage relative to fixed (+0.5 pp and +0.9 pp respectively). In hard, gated has the lowest coverage (0.268 vs 0.266 fixed) because the higher victim drift means UAVs frequently cross the confidence threshold even during search.

---

### 7.6 Table 4: Ablation over α (Hard Scenario, 30 Seeds)

| α | Tracking | Miss | First Det. | Coverage |
|---|---|---|---|---|
| 0.00 | 0.905 ± 0.003 | 0.095 | 3.37 ± 0.72 | 0.295 ± 0.034 |
| 0.10 | 0.909 ± 0.003 | 0.091 | 2.67 ± 0.76 | 0.270 ± 0.020 |
| 0.25 | 0.915 ± 0.003 | 0.085 | 2.30 ± 0.65 | 0.266 ± 0.006 |
| 0.40 | 0.918 ± 0.003 | 0.082 | 2.07 ± 0.45 | 0.274 ± 0.006 |

Tracking and first-detection improve monotonically with α. The step from α=0.1 to α=0.25 gives the largest single improvement in first-detection (2.67 → 2.30 steps). The step from α=0.25 to α=0.4 adds only 0.23 steps and 0.003 tracking ratio — within one standard deviation.

---

### 7.7 Soft Gate vs Binary Gate (30 Seeds)
The soft gate achieves the best tracking on medium (0.890 vs 0.876 fixed, 0.887 binary), recovers hard-scenario coverage relative to binary gate (0.274 vs 0.268), and improves hard first-detection over binary gate (3.13 vs 3.37 steps). It does not match fixed residual on first-detection (3.13 vs 2.30), because the sigmoid still suppresses the residual during early search when confidence is low.

The soft gate's main advantage over binary gate is smoother activation under high drift: rather than switching abruptly at τ, it scales the correction proportionally to confidence, reducing over-correction in the hard scenario.

---

### 7.8 Reading the Pattern — What the Numbers Tell You

**Pattern 1: Tracking always improves.**
The network learned to maintain closer contact with victims. This is the primary contribution.

**Pattern 2: First detection always improves.**
The network learned to respond faster when a victim is first spotted — likely by learning to redirect nearby drones more aggressively at the moment of first contact.

**Pattern 3: Gains are larger in harder scenarios.**
Hard first detection: 3.37 → 2.30 (31.8% faster). The harder the scenario, the more the network helps. This makes sense — the expert's fixed weights are most suboptimal when conditions are most challenging.

**One inconsistency: fixed residual tracking drops on medium (0.876 vs 0.879 hybrid).**
The gated variant (0.887) and soft gate (0.890) both beat the hybrid on medium. This suggests the binary and soft gates are better suited to medium difficulty — the gate correctly suppresses the residual during search phases, while the fixed residual applies correction even when it's not helpful.

---

### 7.9 MARL Baselines — Why End-to-End Learning Fails

To validate the central thesis (structured priors dominate end-to-end learning), we trained two standard MARL methods:

**Independent PPO (IPPO):**
- 20,165 parameters (44.8x more than our 450)
- Separate actor-critic per agent, shared weights
- Learning rate sweep: 3e-4, 1e-4, 3e-5
- Best result: 0.505-0.589 tracking ratio (1.65x below ours)

**QMIX (Value Decomposition):**
- 345,978 parameters (770x more than ours)
- 25 discrete actions (8 dirs x 3 speeds + stay)
- 50K replay buffer, 21 episodes/update, gradient clip 0.5
- Best result: 0.000-0.025 tracking ratio (complete failure)

| Method | Params | Default | Medium | Hard |
|---|---|---|---|---|
| QMIX (best LR) | 345,978 | 0.000 | 0.013 | 0.025 |
| IPPO (best LR) | 20,165 | 0.553 | 0.589 | 0.505 |
| Hybrid base | 0 | 0.869 | 0.879 | 0.905 |
| Fixed residual (ours) | 450 | 0.877 | 0.904 | 0.912 |
| Soft gate (ours) | 450 | 0.881 | 0.913 | 0.910 |

The progression QMIX (0.025) < IPPO (0.505) < Hybrid (0.905) < ES residual (0.912) confirms the thesis: end-to-end MARL fails structurally on sparse-reward maritime SAR.

---

### 7.10 Robustness Study 1 — Packet Drop (0-60%)

Event-triggered communication is tested under increasing packet loss:

| Drop Rate | Hybrid | Fixed | Soft |
|---|---|---|---|
| 0% | 0.905 | 0.912 | 0.910 |
| 20% | 0.905 | 0.911 | 0.910 |
| 40% | 0.904 | 0.911 | 0.909 |
| 60% | 0.904 | 0.910 | 0.909 |

Maximum degradation: 0.002 absolute across all methods. Event-triggered communication provides inherent resilience because most decisions are local — messages are helpful but not critical.

---

### 7.11 Robustness Study 2 — Sensor Noise and Actuation Delay

Three noise dimensions tested without retraining:
- **GPS noise:** Gaussian position error (σ = 0-5m)
- **Detection decay:** Exponential probability reduction with distance (rate 0-1.0)
- **Actuation delay:** First-order velocity lag (0-40%)

Combined noise profiles (hard scenario, 30 seeds):

| Profile | GPS σ(m) | Det. Decay | Act. Delay | Hybrid | Fixed | Soft |
|---|---|---|---|---|---|---|
| Ideal | 0 | 0 | 0 | 0.905 | 0.912 | 0.910 |
| Mild | 2 | 0.3 | 0.1 | 0.901 | 0.906 | 0.904 |
| Moderate | 3 | 0.5 | 0.2 | 0.900 | 0.901 | 0.898 |
| Severe | 5 | 1.0 | 0.4 | 0.895 | 0.883 | 0.894 |

All methods maintain tracking above 0.883 under severe combined noise without retraining. The soft gate degrades more gracefully under detection decay because it suppresses corrections when confidence is unreliable.

---

### 7.12 What Changed — Full Version History

| Item | Original | Current (RA-L submission) |
|---|---|---|
| Training iterations | 10 | **100** (pop 24, 7 training seeds) |
| Evaluation seeds | 10 | **30** |
| Residual variants | 1 (fixed only) | **3 (fixed + binary gate + soft gate)** |
| MARL baselines | None | **IPPO (20K params) + QMIX (346K params)** |
| Robustness studies | None | **Packet drop (0-60%) + Sensor noise** |
| Sensor model | Binary detection | **+ probabilistic decay at eval time** |
| Actuation model | Instant velocity | **+ first-order lag at eval time** |
| Statistical testing | Mean ± std only | **Paired t-tests with p-values** |
| Paper format | 5-page report | **IEEE RA-L journal format** |

---

### 7.13 Remaining Limitations
1. **Sensor model during training** — noise/decay tested at evaluation only; full integration during training is future work.
2. **No real vision stack** — detection is distance-based, not camera/radar.
3. **Custom simulator only** — no comparison against established maritime benchmarks.
4. **Static number of UAVs** — does not test dynamic agent addition/removal.

---

### 7.14 The Complete Story in One Paragraph
We built a maritime drone swarm simulator with drifting victims, local belief maps, and sparse event-triggered communication. A 450-parameter MLP trained via evolutionary strategy (population 24, 100 iterations, 7 training seeds) learns bounded corrections on top of a hybrid frontier-belief base controller. Three residual variants were evaluated: fixed-scale, binary-gated, and soft-gated. The soft-gated variant achieves the best tracking on medium (0.913 vs 0.879 hybrid, p < 0.001) while the fixed residual dominates on first-detection speed (3.37 → 2.37 steps on hard, p < 0.001). Compared against IPPO (20,165 params) and QMIX (346K params), our 450-parameter residual achieves 1.65x higher tracking — confirming that structured priors dominate end-to-end RL on sparse-reward maritime SAR. Robustness studies show all methods maintain performance under 60% packet drop and severe combined sensor noise without retraining.

---

## STEP 8: Files and How to Reproduce

### Repository Structure
```
paper/
  main.tex                      — IEEE RA-L submission manuscript
  figures/                      — all paper figures
  fig_visualization.pdf         — trajectory visualization
  fig_training_curve.pdf        — ES training convergence

code/
  # Core training
  train_residual_mlp.py         — train fixed residual (ES, 100 iterations)
  train_gated_residual.py       — train binary-gated residual
  train_soft_gated_residual.py  — train soft-gated residual
  train_ippo.py                 — Independent PPO baseline
  train_qmix_fair.py            — QMIX baseline (fair setup)

  # Evaluation
  compare_all_methods.py        — 30-seed evaluation of all methods
  run_ablation_study.py         — alpha ablation {0, 0.1, 0.25, 0.4}
  run_ippo_lr_sensitivity.py    — IPPO learning rate sweep
  evaluate_packet_drop.py       — robustness under 0-60% packet drop
  evaluate_sensor_noise.py      — robustness under sensor/actuation noise
  multi_seed_evaluation.py      — full 30-seed benchmark
  compute_pvalues.py            — paired t-tests for significance

  # Utilities
  evaluation_utils.py           — shared evaluation utilities
  experiment_config.py          — seeds, scenarios, hyperparameters
  requirements.txt              — numpy, torch (for IPPO/QMIX)

  # Simulator
  src/swarm_sim/
    simulator.py                — MaritimeSwarmSimulator (sensor noise, actuation delay)
    env_interface.py            — SwarmLearningEnv (residual modes, GPS noise)
    config.py                   — SimulationConfig dataclass
    scenarios.py                — scenario configurations
    mlp_policy.py               — 25→16→2 MLP forward pass
    policies.py                 — 4 handcrafted policies
    belief.py                   — 3-stage belief update
    coverage.py                 — coverage grid decay
    render.py                   — visualization

outputs/
  # Trained weights
  residual_mlp_theta.npy        — trained fixed residual weights
  gated_residual_mlp_theta.npy  — trained binary gate weights
  soft_gated_mlp_theta.npy      — trained soft gate weights
  pure_mlp_theta.npy            — pure MLP baseline weights
  ippo_params.npy               — trained IPPO weights

  # Evaluation results
  multi_seed_summary.json       — 30-seed main results
  pvalue_analysis.json          — statistical significance
  ippo_lr_sensitivity.json      — IPPO LR sweep results
  qmix_fair_results.json        — QMIX evaluation results
  packet_drop_robustness.json   — packet drop robustness
  sensor_noise_robustness.json  — sensor noise robustness
```

### Reproduce Results
```bash
cd code
pip install -r requirements.txt

# 30-seed evaluation using pre-trained weights (~5 min)
python3 compare_all_methods.py

# Re-train from scratch (~12 min each)
python3 train_residual_mlp.py
python3 train_gated_residual.py
python3 train_soft_gated_residual.py

# MARL baselines
python3 train_ippo.py
python3 run_ippo_lr_sensitivity.py
python3 train_qmix_fair.py

# Robustness studies
python3 evaluate_packet_drop.py
python3 evaluate_sensor_noise.py

# Ablation and statistics
python3 run_ablation_study.py
python3 compute_pvalues.py
```

### Key Configuration (experiment_config.py)
```python
FROZEN_RESIDUAL_SCALE = 0.25      # alpha
PAPER_BASE_POLICY = "hybrid_frontier_belief"
SCENARIOS = ["default", "medium", "hard"]
TRAIN_SEEDS = [3, 7, 11, 15, 19, 23, 27]  # 7 seeds for training
PAPER_SEEDS = [3, 7, 11, ..., 119]         # 30 seeds for evaluation
```
Training uses 7 seeds. Evaluation uses 30. This prevents overfitting to specific random conditions.
