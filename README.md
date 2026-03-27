# 6×6 Checkers (`mycheckersenv.py`)

A two-player 6×6 checkers environment built on [PettingZoo's AECEnv](https://pettingzoo.farama.org/api/aec/) (Agent-Environment Cycle API).

```python
from mycheckersenv import CheckersEnv
env = CheckersEnv(render_mode="human")
```

| Attribute             | Details                                      |
|-----------------------|----------------------------------------------|
| Agents                | 2                                            |
| Agent Names           | `player_0`, `player_1`                       |
| Action Space          | `Discrete(144)`                              |
| Action Values         | `[0, 143]`                                   |
| Observation Shape     | `(36,)`                                      |
| Observation Values    | `[-2, 2]`                                    |
| Action Mask Shape     | `(144,)`                                     |
| Parallel API          | No                                           |
| Manual Control        | No                                           |

---

## Observation Space

Each agent's observation is a dictionary with two keys:

```python
spaces.Dict({
    "observation": spaces.Box(low=-2, high=2, shape=(36,), dtype=np.int8),
    "action_mask": spaces.Box(low=0,  high=1,  shape=(144,), dtype=np.int8),
})
```

### `"observation"` — shape `(36,)`

The 6×6 board flattened into a 1D array of 36 integers. Each cell encodes the piece at that square:

| Value | Meaning            |
|-------|--------------------|
| `0`   | Empty square       |
| `1`   | player_0's piece   |
| `2`   | player_0's king    |
| `-1`  | player_1's piece   |
| `-2`  | player_1's king    |

**Perspective flip:** player_1's observation is negated (`obs = -obs`) so that both agents always see their own pieces as positive values. This allows a single policy network to play as either color.

player_0 starts at the bottom (rows 4–5) and moves **up** (decreasing row index).
player_1 starts at the top (rows 0–1) and moves **down** (increasing row index).

### `"action_mask"` — shape `(144,)`

A binary array of length 144. `mask[i] = 1` means action `i` is legal in the current state; `mask[i] = 0` means it is not. Agents must only select actions where the mask is `1`.

---

## Action Space

Each agent's action is a single integer in `[0, 143]` — an index into a fixed lookup table (`action_map`) that maps every action ID to a `(from_pos, to_pos)` board coordinate pair.

### Why 144?

There are **18 dark squares** on a 6×6 board. From any square, a piece could theoretically make up to **8 diagonal moves**: 4 single-step (±1 row, ±1 col) and 4 two-step jumps (±2 row, ±2 col).

```
18 squares × 8 moves = 144 action IDs
```

Most actions are illegal at any moment. The action mask (above) indicates which are legal.

### Action Map

`env.action_map` is a pre-built dictionary mapping action IDs to coordinate pairs:

```python
env.action_map[action_id]  # returns ((from_row, from_col), (to_row, to_col))
```

---

## Rewards

| Outcome      | player_0 | player_1 |
|--------------|----------|----------|
| player_0 wins | `+1`   | `-1`     |
| player_1 wins | `-1`   | `+1`     |
| Non-terminal step | `0` | `0`   |

Rewards are **sparse** — agents only receive feedback at game end.

---

## Termination Conditions

An episode terminates when either of the following is true:

1. **No pieces remaining** — a player has all pieces captured.
2. **No legal moves** — a player has pieces but no legal move available on their turn.

The player who causes either condition wins. Draws are not part of the standard rules but can be enforced externally via a max-step limit (set to 200 in the training script).

---

## Game Rules

- **Mandatory jumps**: if any capture (jump) is available, `_get_legal_moves` returns only jumps — the agent must capture.
- **King promotion**: a piece reaching the opponent's back row (row 0 for player_0, row 5 for player_1) becomes a king (value `±2`) and may move in all 4 diagonal directions.

---

## Usage

```python
from mycheckersenv import CheckersEnv
import numpy as np

env = CheckersEnv(render_mode="human")
env.reset()

for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()

    if terminated or truncated:
        env.step(None)
        continue

    action_mask = obs["action_mask"]
    legal_actions = np.where(action_mask == 1)[0]
    action = np.random.choice(legal_actions)  # replace with your policy

    env.step(action)
```

---

## Quick Start: Training and Sample Run

```bash
# Activate environment
source myenv/bin/activate

# Train the agent (saves trained_agent.pth)
python myrunner.py

# Watch a sample self-play game with the trained agent
python sample_run.py
```

See `IMPLEMENTATION.md` for full details on the Actor-Critic agent and training loop.
