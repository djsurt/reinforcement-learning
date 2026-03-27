# Runner Explained: `myrunner.py`

This document explains how the training loop works — who the agent plays against, how updates happen mid-game, and why training is structured in two phases.

---

## The Big Picture

The runner does one thing: **train the agent by playing games**. It has three concerns:

1. Who does the agent play against?
2. How does the agent learn after each step?
3. How do we measure progress?

---

## Part 1: Two Training Phases

```python
PRETRAIN_EPISODES = 15000   # Phase 1: vs random opponent
SELFPLAY_EPISODES = 15000   # Phase 2: vs past versions of itself
```

### Phase 1: Pre-train vs Random (Episodes 1–15,000)

The agent plays against a random opponent that picks any legal move uniformly.

**Why start here?** If you begin self-play from scratch, both sides are equally terrible — neither knows anything, so neither gives useful training signal. Pre-training teaches basic strategy first:
- Capture opponent pieces (reward shaping gives +0.2)
- Don't lose pieces (reward shaping gives -0.2)
- Win games (+1)

### Phase 2: Self-Play (Episodes 15,001–30,000)

The agent plays against snapshots of its past self.

```python
if episode == PRETRAIN_EPISODES + 1:
    # Lower policy lr for self-play (finer tuning)
    for pg in agent.theta_optimizer.param_groups:
        pg["lr"] = 5e-5       # down from 1e-4
    opponent_pool = [agent.clone()]
```

The learning rate is reduced for self-play because the agent is already competent — we want fine-tuning, not big jumps.

---

## Part 2: `play_episode` — The Core Training Loop

This function plays one full game and calls `agent.update()` after every single agent step. This is the "one-step" part of one-step actor-critic.

### Step 1: Setup

```python
env.reset()
agent_seat = random.choice(["player_0", "player_1"])
I = 1.0
prev_obs = None
prev_mask = None
prev_action = None
prev_material = None
```

- `agent_seat`: randomly assigns the learning agent to either side. This prevents the agent from specializing in only playing as one color.
- `I = 1.0`: the discount accumulator starts fresh each episode (textbook: I ← 1)
- `prev_*`: these hold the "pending" previous transition, explained below.

---

### Step 2: The AEC Turn Loop

```python
for current_agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
```

PettingZoo's AEC API alternates turns: player_0, player_1, player_0, player_1, ...

`env.last()` returns:
- `obs`: dict with `"observation"` (board) and `"action_mask"` (legal actions)
- `reward`: the reward accumulated since this agent's last turn
- `terminated`: True if the game is over

---

### Step 3: The Delayed Update Problem

This is the trickiest part. The textbook says:

```
Take action A in state S
Observe S' and R
Update using (S, A, R, S')
```

But in a two-player game, after the agent acts, **the opponent gets a turn before S' is observed**:

```
Agent's turn:    board = S  → agent takes action A → board changes
Opponent's turn:            → opponent acts        → board changes = S'
Agent's turn:    board = S' → NOW we can form (S, A, R, S') and update
```

So the update is **delayed by one turn**. The agent always updates using the *previous* step's data when it arrives at the *current* step:

```python
if current_agent == agent_seat:
    if prev_obs is not None:
        # S' has arrived — complete the update for the previous step
        material_now = _get_material(env.board, agent_seat)
        shaped_r = REWARD_SHAPING_SCALE * (material_now - prev_material)
        agent.update(prev_obs, prev_mask, prev_action, shaped_r, board_obs, False, I)
        I *= agent.gamma       # I <- gamma * I

    # Now take a new action for the current step
    action = agent.select_action(board_obs, action_mask)

    # Store this step as "pending" — will be completed next turn
    prev_obs = board_obs.copy()
    prev_mask = action_mask.copy()
    prev_action = action
    prev_material = _get_material(env.board, agent_seat)
```

**Visual:**
```
Turn 1 (agent):    take action A0 in S0   → store prev_obs=S0, prev_action=A0
Turn 2 (opponent): opponent acts          → board changes
Turn 3 (agent):    arrived at S2          → update(S0, A0, shaped_r, S2) ← delayed update
                                          → take action A2 in S2
                                          → store prev_obs=S2, prev_action=A2
Turn 4 (opponent): opponent acts
Turn 5 (agent):    arrived at S4          → update(S2, A2, shaped_r, S4)
                   ...
```

---

### Step 4: Terminal States

When the game ends, `terminated=True` and `reward` holds the final +1 or -1:

```python
if terminated or truncated:
    if current_agent == agent_seat and prev_obs is not None:
        material_now = _get_material(env.board, agent_seat)
        shaped_r = REWARD_SHAPING_SCALE * (material_now - prev_material)
        R = reward + shaped_r                                 # terminal reward + shaped reward
        agent.update(prev_obs, prev_mask, prev_action, R, None, True, I)

        winner = "agent" if reward > 0 else "opponent" if reward < 0 else None

    env.step(None)   # PettingZoo requires this to acknowledge game-over
    continue
```

The `None` passed as `next_obs` to `agent.update()` signals "terminal state." The agent code handles this:

```python
if done:
    v_s_next = 0.0   # textbook: v̂(terminal, w) = 0
```

### Step 5: Max Step Truncation

```python
if step_count >= MAX_STEPS_PER_EPISODE:
    if prev_obs is not None:
        agent.update(prev_obs, prev_mask, prev_action, shaped_r, None, True, I)
    break
```

If a game drags on past 200 steps (common early in training when both sides are weak), we cut it off as a draw. The pending update is flushed with `done=True`.

---

## Part 3: Reward Shaping

The environment only gives reward at the very end (+1 win, -1 loss). Learning from one reward per game is extremely slow — a 50-move game has 50 steps, only one of which carries a signal.

To speed this up, we add an **intermediate shaped reward** based on material (piece count):

```python
def _get_material(board, agent_name):
    p0_pieces = np.sum(board == 1)
    p0_kings  = np.sum(board == 2)   # kings worth 2x
    p1_pieces = np.sum(board == -1)
    p1_kings  = np.sum(board == -2)
    if agent_name == "player_0":
        return (p0_pieces + 2.0 * p0_kings) - (p1_pieces + 2.0 * p1_kings)
    else:
        return (p1_pieces + 2.0 * p1_kings) - (p0_pieces + 2.0 * p0_kings)
```

The score is always from the learning agent's perspective: my material minus opponent's material.

The shaped reward between turns:

```python
shaped_r = 0.2 * (material_now - prev_material)
```

| Event | Material change | Shaped reward |
|-------|----------------|---------------|
| Capture opponent piece | +1 | +0.2 |
| Capture opponent king | +2 | +0.4 |
| Lose a piece | -1 | -0.2 |
| Promote to king | 0 (already counted) | 0 |
| No captures | 0 | 0 |

The 0.2 scale keeps shaped rewards small so the final +1/-1 game reward still dominates. The agent learns from feedback every turn instead of waiting for game end.

**Important:** `_get_material` is called AFTER `env.step()` so captures are already reflected on the board.

---

## Part 4: The Self-Play Opponent Pool

```python
opponent_pool = [agent.clone()]   # start with one copy of the current agent
```

Every 100 episodes during phase 2, we snapshot the current agent:

```python
if episode > PRETRAIN_EPISODES and episode % SNAPSHOT_INTERVAL == 0:
    opponent_pool.append(agent.clone())
    if len(opponent_pool) > MAX_POOL_SIZE:
        opponent_pool.pop(0)    # remove oldest snapshot
```

The pool holds at most 10 past versions (a sliding window of the agent's history).

### Opponent Selection

```python
def select_opponent(opponent_pool):
    if random.random() < 0.8:
        return opponent_pool[-1]        # 80% → latest snapshot (strongest)
    return random.choice(opponent_pool) # 20% → random old snapshot
```

And within the training loop:

```python
if random.random() < 0.3:
    opponent_fn = random_opponent       # 30% → fully random
else:
    opp = select_opponent(opponent_pool)
    opponent_fn = make_agent_opponent(opp)  # 70% → snapshot
```

**Why not always use the latest snapshot?**

Strategy cycling: agent A learns to beat B, B learns to beat C, C beats A — the policy oscillates. Old snapshots force the agent to be robust against diverse strategies.

**Why 30% random during self-play?**

Without it, the agent "forgets" how to beat random opponents while specializing in beating itself. Mixing in 30% random keeps basic skills sharp.

---

## Part 5: Evaluation

```python
def evaluate_vs_random(agent, num_games=50):
    wins = 0
    for _ in range(num_games):
        env = CheckersEnv()
        env.reset()
        agent_seat = random.choice(["player_0", "player_1"])
        ...
        for current_agent in env.agent_iter():
            if current_agent == agent_seat:
                action = agent.select_action(board_obs, action_mask)
            else:
                action = np.random.choice(legal)   # no update, just random
            env.step(action)
    return wins / num_games
```

Every 2,500 episodes, we run 50 games with **no calls to `agent.update()`** — pure evaluation. This gives a clean measurement of how good the agent actually is, uncontaminated by ongoing learning.

The result is printed alongside training win rate and pool size:

```
[pretrain] Ep   2500 | vs Random: 58.0% | Train WR: 54.0% | Pool: 0
[pretrain] Ep   5000 | vs Random: 71.0% | Train WR: 65.0% | Pool: 0
...
[selfplay] Ep  17500 | vs Random: 88.0% | Train WR: 61.0% | Pool: 15
```

---

## Part 6: `make_agent_opponent` — Why a Wrapper?

```python
def make_agent_opponent(opponent_agent):
    def opponent_fn(obs, action_mask):
        return opponent_agent.select_action(obs, action_mask)
    return opponent_fn
```

The training loop expects `opponent_fn(obs, mask)` — a callable that takes board state and mask, returns an action. `random_opponent` already has this signature. But `CheckersAgent.select_action` needs `self`, so we wrap it in a closure to match the interface.

This means `random_opponent` and agent opponents are interchangeable in the training loop — the loop doesn't need to know which type it's dealing with.

---

## Full Training Flow

```
train()
    │
    ├─ Phase 1 (episodes 1–15,000)
    │       opponent = random_opponent
    │       play_episode(env, agent, random_opponent)
    │           └─ for each step:
    │                  select_action → env.step → update
    │
    ├─ Phase 2 (episodes 15,001–30,000)
    │       lr reduced to 5e-5
    │       opponent_pool initialized
    │       │
    │       ├─ 30% → opponent = random_opponent
    │       └─ 70% → opponent = snapshot from pool
    │           └─ 80% latest, 20% random old
    │
    │       every 100 episodes: add clone to pool (max 10)
    │
    ├─ every 2,500 episodes: evaluate_vs_random (50 games, no training)
    │
    └─ end: save weights to trained_agent.pth
```

---

## Key Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `PRETRAIN_EPISODES` | 15,000 | How long to train vs random before self-play |
| `SELFPLAY_EPISODES` | 15,000 | How long self-play runs |
| `MAX_STEPS_PER_EPISODE` | 200 | Cut off long games as a draw |
| `SNAPSHOT_INTERVAL` | 100 | How often to save a snapshot to the pool |
| `MAX_POOL_SIZE` | 10 | How many past versions to keep |
| `EVAL_INTERVAL` | 2,500 | How often to evaluate vs random |
| `EVAL_GAMES` | 50 | How many evaluation games per checkpoint |
| `REWARD_SHAPING_SCALE` | 0.2 | How much to weight intermediate material rewards |
