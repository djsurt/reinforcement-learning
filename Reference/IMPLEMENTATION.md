# Implementation Details

This document covers the Actor-Critic agent (`myagent.py`) and the self-play training loop (`myrunner.py`).

---

# Actor-Critic Agent (`myagent.py`)

## Overview

The agent implements the **One-step Actor-Critic (episodic)** algorithm from Sutton & Barto. It uses two separate sets of parameters, with PyTorch providing automatic differentiation:

- **θ** (policy parameters) — the *actor*, defines π(a|s, θ) which decides which action to take
- **w** (value weights) — the *critic*, defines v̂(s, w) which estimates how good the current board position is

Both use a single hidden layer for expressiveness (the textbook requires "differentiable parameterizations" — a hidden layer with ReLU is one such parameterization):

```
Board state (36 numbers)
        │
        ├──────────────────────┐
        ▼                      ▼
┌──────────────────┐   ┌──────────────────┐
│  Policy: π(a|s,θ)│   │  Value: v̂(s,w)  │
│                  │   │                  │
│  θ_W1 (36→128)  │   │  w_W1 (36→128)  │
│  ReLU            │   │  ReLU            │
│  θ_W2 (128→144) │   │  w_W2 (128→1)   │
│                  │   │                  │
│  144 logits      │   │  scalar value    │
└──────────────────┘   └──────────────────┘
```

θ and w are stored as plain PyTorch `nn.Parameter` tensors (not as neural network classes). The updates follow the textbook equations exactly, with Adam optimizers handling the adaptive step sizes α^θ and α^w.

---

## Action Masking

The policy outputs 144 raw logits — one per action. Most actions are illegal. Before sampling, illegal actions are masked out:

```python
masked_logits = logits + (action_mask - 1.0) * 1e9
```

When `mask = 0` (illegal): logit becomes `logit - 1e9` → effectively `-∞` after softmax → zero probability.
When `mask = 1` (legal): logit is unchanged.

This is done **before** the softmax. Masking after softmax and renormalizing is numerically unstable. This way, the agent can never accidentally pick an illegal move.

---

## The One-step TD Update

Unlike Monte Carlo methods that wait until the end of an episode, this algorithm updates **after every single step** using the one-step TD error:

```
δ = R + γ · v̂(S', w) - v̂(S, w)
```

Where:
- `R` = reward received after taking action A in state S
- `γ` = discount factor (0.99)
- `v̂(S', w)` = critic's estimate of the next state (0 if S' is terminal)
- `v̂(S, w)` = critic's estimate of the current state

**Intuition:** δ is the "surprise" — the difference between what actually happened (`R + γ · v̂(S')`) and what the critic predicted (`v̂(S)`). If δ > 0, things went better than expected; if δ < 0, worse.

---

## The Update Rules

At each step, two updates happen:

### 1. Critic update (value network)

```
w ← w + α^w · δ · ∇v̂(S, w)
```

In PyTorch, we minimize `loss = -δ · v̂(S, w)`. The optimizer does `w -= lr · ∇loss = w -= lr · (-δ · ∇v̂)`, which gives `w += α^w · δ · ∇v̂`.

### 2. Actor update (policy network)

```
θ ← θ + α^θ · I · δ · ∇ln π(A|S, θ)
```

- If δ > 0 (better than expected): increase probability of action A
- If δ < 0 (worse than expected): decrease probability of action A
- **I** is a discount accumulator (starts at 1, decays by γ each step)

In PyTorch, we minimize `loss = -I · δ · ln π(A|S, θ)`.

### 3. Discount accumulator

```
I ← γ · I
```

I starts at 1.0 at the beginning of each episode and shrinks each step. Actions taken early in the game (when I is large) get stronger updates than actions taken late.

---

## Hyperparameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Policy learning rate | α^θ | 1e-4 (Adam), reduced to 5e-5 in self-play |
| Value learning rate | α^w | 1e-3 (Adam) |
| Discount factor | γ | 0.99 |
| Hidden layer size | — | 128 |
| Optimizer | — | Adam |

---
---

# Self-Play Training (`myrunner.py`)

## Overview

Training happens in two phases. The agent starts by learning from a simple opponent (random), then refines its play by competing against past versions of itself.

```
Phase 1: Pre-train vs Random (15,000 episodes)
    → Agent learns basic strategy: capture pieces, avoid losing pieces

Phase 2: Self-Play (15,000 episodes)
    → Agent plays against snapshots of itself
    → Opponent pool keeps past versions for diversity
    → Agent learns to defeat increasingly strong opponents
```

---

## Full Training Flow

```
for each episode:
    │
    ├─ Phase 1? → opponent = random agent
    ├─ Phase 2? → opponent = 30% random, 70% snapshot from pool
    │
    ▼
play_episode(env, agent, opponent)
    │
    ├── env.reset()
    ├── randomly assign agent to player_0 or player_1
    ├── I ← 1.0  (reset discount accumulator)
    │
    └── for current_agent in env.agent_iter():
            │
            ├── obs, reward, terminated = env.last()
            │
            ├── if terminated:
            │       R = terminal_reward + shaped_reward
            │       agent.update(prev_S, prev_A, R, S'=None, done=True, I)
            │       env.step(None)
            │
            ├── if learning agent's turn:
            │       if prev_S exists:
            │           R = shaped_reward (Δmaterial since last turn)
            │           agent.update(prev_S, prev_A, R, S'=current_obs, done=False, I)
            │           I ← γ * I
            │       action = agent.select_action(obs, mask)
            │       prev_S ← obs, prev_A ← action
            │       env.step(action)
            │
            └── if opponent's turn:
                    action = opponent(obs, mask)
                    env.step(action)
    │
    ▼
every 100 episodes (phase 2 only):
    └── clone agent → add to opponent pool (max 10 snapshots)
```

---

## The Delayed Update

In a two-player game, the agent can't update immediately after acting — the opponent moves before S' is observed. So the update is deferred until the agent's next turn:

```
Agent's turn:    board = S  → take action A → store (S, A)
Opponent's turn:            → opponent acts
Agent's turn:    board = S' → update(S, A, R, S') ← delayed
```

---

## Reward Shaping

Since the environment only gives reward at game end (sparse), training is sped up with an intermediate **shaped reward**:

```python
shaped_reward = 0.2 × (material_after_move - material_before_move)
```

Where material = `own_pieces + 2 × own_kings - opponent_pieces - 2 × opponent_kings`.

Kings are worth 2× because they can move in all 4 directions. The shaped reward gives immediate feedback mid-game while the terminal reward (+1/-1) still dominates.

---

## Self-Play Opponent Pool

```
opponent_pool = [snapshot_0, ..., snapshot_N]  (max 10)

During self-play (Phase 2):
    30% → random agent        (prevents forgetting basic skills)
    70% → from pool:
        80% → latest snapshot (strongest version of self)
        20% → random old snapshot (diversity, prevents strategy cycling)
```

**Strategy cycling** occurs when agent A learns to beat B, B beats C, and C beats A — the policy oscillates. Old snapshots break these cycles.

---

## Perspective Sharing

Both `player_0` and `player_1` use the **same network weights**. This works because the environment negates `player_1`'s observation, so both always see their own pieces as positive. Every game produces training experience from both perspectives, effectively doubling the training data.
