# Actor-Critic Explained: `myagent.py`

This document walks through every part of the actor-critic agent code so you can explain it confidently to your professor.

---

## The Big Idea

Imagine you are coaching a chess player who has never played before. You have two jobs:

1. **Actor** — tells the player which move to make ("go here")
2. **Critic** — tells the player how good the current position is ("you are up a rook, this is +3")

The actor makes decisions. The critic evaluates positions. They train each other: the actor gets better by listening to the critic's feedback, and the critic gets better by comparing its predictions to what actually happened.

In code, the actor is controlled by parameters **θ** (theta) and the critic by parameters **w**.

---

## The Textbook Algorithm

The exact algorithm from Sutton & Barto (the textbook):

```
Input:  a differentiable policy π(a|s, θ)
        a differentiable state-value function v̂(s, w)

Parameters: step sizes α^θ > 0, α^w > 0, discount γ

Initialize θ and w (small random values)

Loop forever (for each episode):
    I ← 1
    S ← initial state

    Loop for each step of episode:
        A ~ π(·|S, θ)               ← sample action from policy
        Take action A, observe R, S'
        δ ← R + γ·v̂(S', w) - v̂(S, w)   ← TD error
        w ← w + α^w · δ · ∇v̂(S, w)     ← update critic
        θ ← θ + α^θ · I · δ · ∇ln π(A|S, θ)  ← update actor
        I ← γ · I                   ← decay discount
        S ← S'
    Until S is terminal
```

Every line of `myagent.py` maps directly to one part of this algorithm.

---

## What are θ and w?

These are just **numbers** — specifically, matrices and vectors of floating-point numbers that define the shape of two mathematical functions.

### θ (theta) — the Policy Parameters

θ defines the function **π(a|s, θ)**, which answers: "given the board state s, what is the probability of taking each action a?"

In code, θ consists of four matrices:
```python
self.theta_W1  # shape (128, 36)  — weights for hidden layer
self.theta_b1  # shape (128,)     — biases for hidden layer
self.theta_W2  # shape (144, 36)  — weights for output layer
self.theta_b2  # shape (144,)     — biases for output layer
```

The computation is:
```
hidden   = ReLU(s · theta_W1ᵀ + theta_b1)   # (36,) → (128,)
logits   = hidden · theta_W2ᵀ + theta_b2    # (128,) → (144,)
```

The 144 output numbers are called **logits** — one per possible action. They are not probabilities yet (they can be any number). We convert them to probabilities using softmax.

### w — the Value Weights

w defines the function **v̂(s, w)**, which answers: "given the board state s, how good is this position?"

```python
self.w_W1  # shape (128, 36)
self.w_b1  # shape (128,)
self.w_W2  # shape (1, 128)
self.w_b2  # shape (1,)
```

The computation is:
```
hidden = ReLU(s · w_W1ᵀ + w_b1)   # (36,) → (128,)
value  = hidden · w_W2ᵀ + w_b2    # (128,) → (1,)
```

The single output number is the estimated value — how good this state is, on a scale that the agent learns over time.

---

## Why `nn.Parameter`?

```python
self.theta_W1 = nn.Parameter(torch.randn(hidden, 36) * 0.01)
```

`nn.Parameter` is PyTorch's way of saying "this tensor needs gradients and is a leaf node in the computation graph."

- **Leaf node** means: this is where gradients stop. PyTorch computes how much each parameter contributed to the loss, then the optimizer adjusts it.
- Without `nn.Parameter`, PyTorch's Adam optimizer would refuse to update the tensor (it would say "can't optimize a non-leaf tensor").

The `* 0.01` initializes weights very close to zero. If weights start too large, the network outputs extreme values, making early training chaotic.

---

## Why are there Two Separate Optimizers?

```python
self.theta_optimizer = torch.optim.Adam([theta_W1, theta_b1, theta_W2, theta_b2], lr=1e-4)
self.w_optimizer     = torch.optim.Adam([w_W1, w_b1, w_W2, w_b2], lr=1e-3)
```

The textbook says α^θ and α^w can be different. Here:
- **α^w = 1e-3** (larger): the critic learns faster. It needs to build a good value estimate quickly so the actor has useful feedback.
- **α^θ = 1e-4** (smaller): the actor learns slower. Policy changes are permanent — if the actor moves too fast in the wrong direction, recovery is slow.

They are separate optimizers because they update different parameters. Adam also keeps a running average of past gradients per-parameter, which means each optimizer needs its own internal state.

---

## Action Masking

The board has 144 possible action IDs, but at any moment only a few are legal (maybe 3–8). If the agent could pick any of the 144, it would frequently pick illegal moves — no learning would happen.

```python
masked_logits = logits + (mask_t - 1.0) * 1e9
```

Step by step:
- `mask_t` is a vector of 144 values: 1 if legal, 0 if illegal
- `(mask_t - 1.0)` becomes: 0 if legal, -1 if illegal
- `* 1e9` becomes: 0 if legal, -1,000,000,000 if illegal
- Adding to logits: legal actions unchanged, illegal actions get logit - 1e9

When softmax converts logits to probabilities:
```
P(illegal action) = exp(-1e9) / Z ≈ 0   (effectively zero)
P(legal action)   = exp(logit) / Z       (normal probability)
```

The agent literally cannot pick an illegal move — their probability becomes zero.

**Why before softmax?** Masking after softmax and renormalizing is numerically unstable (dividing by a sum that might include floating-point errors). Doing it before is clean and exact.

---

## The `update` Method — The Heart of the Algorithm

This is called once after every single agent step. It implements the three textbook update equations.

```python
def update(self, obs, action_mask, action, reward, next_obs, done, I):
```

The arguments are:
| Argument | Textbook symbol | Meaning |
|----------|----------------|---------|
| `obs` | S | Board state before this action |
| `action_mask` | — | Which of 144 actions were legal |
| `action` | A | The action that was taken |
| `reward` | R | Reward received after action |
| `next_obs` | S' | Board state after action (None if terminal) |
| `done` | — | Whether S' is terminal |
| `I` | I | Discount accumulator |

### Step 1: Compute v̂(S, w)

```python
obs_t = torch.FloatTensor(obs)
v_s = self._get_value(obs_t)
```

This runs the value network forward on the current state. The result `v_s` is a PyTorch tensor — it has a **computation graph** attached, which is how PyTorch knows how to compute gradients later.

### Step 2: Compute v̂(S', w)

```python
if done:
    v_s_next = 0.0
else:
    with torch.no_grad():
        v_s_next = self._get_value(torch.FloatTensor(next_obs)).item()
```

If the game is over, the textbook says v̂(terminal, w) = 0 — a terminal state has no future value.

`torch.no_grad()` means: don't track gradients for this computation. We only need the number, not a graph. We are treating v̂(S') as a fixed target, not something we want to differentiate through.

### Step 3: Compute δ (TD error)

```python
delta = reward + self.gamma * v_s_next - v_s.item()
```

This is exactly the textbook equation:
```
δ = R + γ · v̂(S', w) - v̂(S, w)
```

`v_s.item()` extracts the plain Python float (detaching it from the graph). We use the plain float in the `delta` formula so that `delta` itself is a plain number — not a tensor.

**What does δ mean?**
- δ > 0: things went better than expected. The critic was too pessimistic. Reward was higher than predicted.
- δ < 0: things went worse than expected. The critic was too optimistic. Reward was lower than predicted.
- δ = 0: prediction was perfect.

### Step 4: Update the Critic (w update)

Textbook equation:
```
w ← w + α^w · δ · ∇v̂(S, w)
```

In code:
```python
value_loss = -delta * v_s
self.w_optimizer.zero_grad()
value_loss.backward()
self.w_optimizer.step()
```

**Why is the loss `-delta * v_s`?**

PyTorch optimizers perform gradient **descent**: `w ← w - α · ∇loss`.

We want gradient **ascent** on `δ · v̂(S, w)`: `w ← w + α · δ · ∇v̂`.

To convert ascent to descent: flip the sign. So `loss = -(δ · v̂) = -delta * v_s`.

Then:
```
w ← w - α · ∇loss
   = w - α · ∇(-delta * v_s)
   = w - α · (-delta · ∇v_s)
   = w + α · delta · ∇v_s       ← matches textbook
```

`backward()` computes `∂loss/∂w` for every w parameter.
`step()` applies: `w ← w - lr · ∂loss/∂w`.

### Step 5: Update the Actor (θ update)

Textbook equation:
```
θ ← θ + α^θ · I · δ · ∇ln π(A|S, θ)
```

In code:
```python
logits = self._get_policy_logits(obs_t)
mask_t = torch.FloatTensor(action_mask)
masked_logits = logits + (mask_t - 1.0) * 1e9
dist = Categorical(logits=masked_logits)
log_prob = dist.log_prob(torch.tensor(action))

policy_loss = -I * delta * log_prob
self.theta_optimizer.zero_grad()
policy_loss.backward()
self.theta_optimizer.step()
```

`dist.log_prob(action)` computes `ln π(A|S, θ)` — the log probability of the action that was actually taken.

**Why `-I * delta * log_prob`?**

Same sign-flip logic as the critic:
```
loss = -(I · δ · ln π(A|S, θ))

∇loss = -(I · δ · ∇ln π(A|S, θ))

θ ← θ - lr · ∇loss
   = θ - lr · (-(I · δ · ∇ln π))
   = θ + α^θ · I · δ · ∇ln π    ← matches textbook
```

**Intuition for this update:**
- If δ > 0 (reward was better than expected): `log_prob` increases → probability of action A increases → agent does this more
- If δ < 0 (reward was worse than expected): `log_prob` decreases → probability of action A decreases → agent does this less

The agent is being told: "when things go better than expected, do more of what you did; when things go worse, do less."

---

## The Discount Accumulator I

I is initialized to 1.0 at the start of each episode and decays by γ = 0.99 each step:

```
Step 1: I = 1.0
Step 2: I = 0.99
Step 3: I = 0.9801
...
```

Its role is in the actor update: `policy_loss = -I * delta * log_prob`

**Why does this matter?**

Actions taken early in the game (when I is close to 1) get **stronger updates** than actions taken late (when I is small). This correctly reflects that early decisions have more long-term impact — choosing to sacrifice a piece on move 2 is more consequential than a move made when only 2 pieces remain.

Without I, all steps would be weighted equally — which is not how discounted returns work mathematically.

---

## How the Computation Graph Works

This is the PyTorch magic that makes gradients automatic.

When you write:
```python
v_s = self._get_value(obs_t)
```

PyTorch doesn't just compute a number — it builds a **graph** of every operation:
```
obs_t → matmul(w_W1) → add(w_b1) → relu → matmul(w_W2) → add(w_b2) → v_s
```

When you call `value_loss.backward()`, PyTorch walks this graph **backwards** and computes `∂loss/∂w_W1`, `∂loss/∂w_b1`, `∂loss/∂w_W2`, `∂loss/∂w_b2` using the chain rule.

The `zero_grad()` call before `backward()` clears any previously computed gradients — without it, gradients would accumulate across multiple calls.

**Why two separate `backward()` calls?**

The value update and policy update are independent. Each has its own graph:
- `value_loss = -delta * v_s` — graph flows through w parameters only
- `policy_loss = -I * delta * log_prob` — graph flows through θ parameters only

Calling `.backward()` twice is fine because we compute them in sequence and clear gradients in between.

---

## The `clone` Method

```python
def clone(self):
    new_agent = CheckersAgent.__new__(CheckersAgent)
    new_agent.gamma = self.gamma
    new_agent.theta_optimizer = None
    new_agent.w_optimizer = None
    for attr in ["theta_W1", "theta_b1", "theta_W2", "theta_b2",
                  "w_W1", "w_b1", "w_W2", "w_b2"]:
        data = getattr(self, attr).data.clone()
        setattr(new_agent, attr, data)
    return new_agent
```

Used to create opponent snapshots for self-play.

Key points:
- `__new__` creates the object without calling `__init__` — so we skip re-creating the random weights
- `.data.clone()` copies the raw tensor values **without the computation graph** — the snapshot doesn't need to learn, just to select actions
- `theta_optimizer = None` — snapshots don't train, so they don't need optimizers

---

## Summary: One Full Step

Here is what happens every time the agent takes one action during training:

```
1. Agent observes board state S (36 numbers)

2. select_action(S, mask):
   - Forward pass through θ: S → hidden (128) → logits (144)
   - Mask illegal actions: illegal logits → -infinity
   - Softmax: logits → probabilities
   - Sample action A from distribution
   → returns A (an integer 0-143)

3. Environment executes A, returns R and S'

4. update(S, mask, A, R, S', done, I):
   a. Forward pass through w:  S  → v̂(S, w)   [with graph]
   b. Forward pass through w:  S' → v̂(S', w)  [no graph, just a number]
   c. δ = R + γ · v̂(S', w) - v̂(S, w)
   d. value_loss = -δ · v̂(S, w)
      → backward() → w_optimizer.step()       [w updated]
   e. Forward pass through θ: S → log_prob(A)  [with graph]
      policy_loss = -I · δ · log_prob(A)
      → backward() → theta_optimizer.step()    [θ updated]

5. I ← γ · I

6. S ← S'  (next turn begins)
```

This repeats for every single step in every episode — online learning, one step at a time. No waiting until the end of the game.

---

## Key Numbers

| Thing | Value | Why |
|-------|-------|-----|
| Hidden layer size | 128 neurons | Enough to learn piece interactions, not too large to train slowly |
| α^θ (policy lr) | 1e-4 | Slow — policy changes are hard to undo |
| α^w (value lr) | 1e-3 | Fast — critic needs to catch up with the real values |
| γ (discount) | 0.99 | Future rewards count almost as much as immediate ones |
| I (initial) | 1.0 | Full weight on first step |
| Masking constant | 1e9 | Large enough that exp(-1e9) ≈ 0 in any float32 context |
