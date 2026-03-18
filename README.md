# Checkers Environment (`checkers.py`)

A 6×6 checkers environment built on [PettingZoo's AECEnv](https://pettingzoo.farama.org/api/aec/) (Agent-Environment Cycle), designed for two-player reinforcement learning.

---

## Overview

The environment follows the **AEC (turn-based) API**: agents take turns one at a time. The two agents are `"player_0"` and `"player_1"`. The game is played on a 6×6 board using only the dark squares (18 total playable squares).

### Board Encoding

The board is a 6×6 NumPy array where each cell holds one of five values:

| Value | Meaning               |
|-------|-----------------------|
| `0`   | Empty square          |
| `1`   | player_0's piece      |
| `2`   | player_0's king       |
| `-1`  | player_1's piece      |
| `-2`  | player_1's king       |

player_0 starts at the bottom (rows 4–5) and moves **up** (decreasing row index).
player_1 starts at the top (rows 0–1) and moves **down** (increasing row index).

---

## Action Space

```python
self.action_spaces = {
    agent: spaces.Discrete(N_ACTIONS)   # N_ACTIONS = 144
}
```

Each agent's action is a **single integer from 0 to 143**. This integer is an *action ID* — a fixed index into a lookup table of every conceivable move on the board.

### Why 144?

There are **18 dark squares** on a 6×6 board. From any square, a piece could theoretically make up to **8 diagonal moves**: 4 single-step (±1 row, ±1 col) and 4 two-step jumps (±2 row, ±2 col).

```
18 squares × 8 moves = 144 action IDs
```

Most of these 144 actions will be illegal at any given moment (out-of-bounds, occupied, wrong direction, etc.). The **action mask** (described below) tells the agent which ones are legal right now.

---

## The `action_map` — Why It Exists

```python
self.action_map = self._build_action_map()
# Example entries:
# {0: ((1,0), (0,1)),   # action 0 = move from (1,0) to (0,1)
#  1: ((1,0), (0,-1)),  # action 1 = move from (1,0) to (0,-1)  [out-of-bounds, always illegal]
#  ...
#  143: ((4,5), (2,3))} # action 143 = jump from (4,5) to (2,3)
```

**This is a fixed, pre-built dictionary that maps every action ID (0–143) to a pair of board coordinates: `(from_pos, to_pos)`.**

The RL agent does not think in terms of coordinates — it thinks in terms of integers. But the board operates on coordinates. The `action_map` is the **translation layer** between them.

The full pipeline for every turn:

```
1. action_map (built once at init)
      action_id (int) → (from_pos, to_pos)

2. _get_legal_moves() (computed each turn)
      board state → list of legal (from_pos, to_pos) pairs

3. _get_action_mask() (bridges steps 1 and 2)
      for each action_id in action_map:
          if (from_pos, to_pos) is in legal_moves → mask[action_id] = 1

4. Agent picks an action_id where mask[action_id] == 1

5. _apply_action() uses action_map to decode the chosen id
      action_id → (from_pos, to_pos) → move on board
```

So `_get_legal_moves` answers "what moves are valid?", and `action_map` answers "what integer does the agent use to select each of those moves?". They work together — neither is sufficient alone.

---

## Observation Space

```python
self.observation_spaces = {
    agent: spaces.Dict({
        "observation": spaces.Box(low=-2, high=2, shape=(36,), dtype=np.int8),
        "action_mask": spaces.Box(low=0,  high=1,  shape=(144,), dtype=np.int8),
    })
}
```

Each agent's observation is a **dictionary with two keys**:

### `"observation"` — shape `(36,)`

The 6×6 board flattened into a 1D array of 36 integers (each in `[-2, 2]`).

- Values use the same encoding as the internal board (`1`=piece, `2`=king, `-1`=opponent piece, etc.)
- **Perspective flip**: player_1's observation is negated (`obs = -obs`) so that **both agents always see their own pieces as positive**. This lets a single policy network work for both players without needing to know which agent it is.

### `"action_mask"` — shape `(144,)`

A binary array of length 144. `mask[i] = 1` means action `i` is legal right now; `mask[i] = 0` means it is not. The agent should only sample from actions where the mask is `1`.

---

## How `observe` Works

```python
def observe(self, agent):
    return self._observe(agent)

def _observe(self, agent):
    obs = self.board.flatten().copy()       # 6×6 → (36,)
    if agent == "player_1":
        obs = -obs                          # flip so my pieces > 0
    mask = self._get_action_mask(agent)     # (144,) binary array
    return {"observation": obs, "action_mask": mask}
```

Step by step:

1. **Flatten the board** — converts the 2D 6×6 grid into a 1D array of 36 values.
2. **Flip perspective for player_1** — negates the array so player_1 sees its pieces as positive (1 or 2) and player_0's pieces as negative. player_0 sees the raw board, no flip needed.
3. **Compute the action mask** — calls `_get_legal_moves`, then cross-references with `action_map` to mark which of the 144 action IDs are currently legal.
4. **Return the dict** — packages both the board state and the mask so the agent has everything it needs.

---

## Key Methods Summary

| Method | Purpose |
|--------|---------|
| `_build_action_map()` | One-time build of all 144 possible (from, to) coordinate pairs indexed by action ID |
| `_init_board()` | Sets up the starting 6×6 board with pieces in rows 0–1 (player_1) and 4–5 (player_0) |
| `reset()` | Resets game state, returns first observation |
| `step(action)` | Applies the agent's chosen action ID, checks for winner, advances turn |
| `observe(agent)` | Returns `{"observation": ..., "action_mask": ...}` for the given agent |
| `_get_legal_moves(agent)` | Returns all valid `(from_pos, to_pos)` moves; **jumps are mandatory** if available |
| `_get_action_mask(agent)` | Converts legal moves into a 144-length binary mask aligned to `action_map` |
| `_apply_action(agent, action)` | Moves the piece, removes jumped pieces, promotes to king if applicable |
| `_check_winner()` | Returns the winning agent if one player has no pieces or no legal moves, else `None` |

---

## Game Rules Implemented

- **Mandatory jumps**: if any jump is available, `_get_legal_moves` returns only jumps (not quiet moves).
- **King promotion**: a piece reaching the opponent's back row (row 0 for player_0, row 5 for player_1) becomes a king (value `±2`) and can move in all 4 diagonal directions.
- **Win conditions**: you win if the opponent has no pieces left, or if it is the opponent's turn and they have no legal moves.
