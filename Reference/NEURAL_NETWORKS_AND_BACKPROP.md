# Neural Networks and Backpropagation: Deep Dive

This document explains how neural networks work in the context of actor-critic reinforcement learning, and how backpropagation allows the network to learn.

---

## Part 1: Neural Networks as Function Approximators

### The Problem: We Need Two Functions

The textbook actor-critic algorithm requires two functions:

1. **π(a|s, θ)** — given board state `s`, output probabilities over actions
2. **v̂(s, w)** — given board state `s`, output a single value estimate

Both must be **differentiable** (we can compute gradients through them) and **parameterized** (they have parameters we can adjust).

The simplest way to build such functions is with neural networks.

---

### Why Neural Networks?

A **linear function** is too simple:

```
output = s · W + b
```

This can only learn linear relationships. "If square 5 has a piece, increase action 17's score by 0.3." But it can't learn interactions: "if squares 5 AND 6 both have pieces, that's a great jumping position."

A **neural network with hidden layers** can learn these interactions.

---

### The Architecture: From Input to Output

The network has three stages:

```
Input layer (36 numbers)
        ↓
Hidden layer (128 numbers)
        ↓
Output layer (144 numbers for policy, 1 number for value)
```

---

## Part 2: How the Policy Network Works (θ)

### The Forward Pass

```python
def _get_policy_logits(self, obs_t):
    hidden = torch.relu(obs_t @ self.theta_W1.T + self.theta_b1)
    return hidden @ self.theta_W2.T + self.theta_b2
```

Step by step:

| Step | Math | Code | Shape |
|------|------|------|-------|
| Input | s | `obs_t` | (36,) |
| Multiply by W1 | s · W1ᵀ | `obs_t @ self.theta_W1.T` | (128,) |
| Add bias | + b1 | `+ self.theta_b1` | (128,) |
| Apply ReLU | ReLU(·) | `torch.relu(...)` | (128,) — some zeros |
| Multiply by W2 | h · W2ᵀ | `hidden @ self.theta_W2.T` | (144,) |
| Add bias | + b2 | `+ self.theta_b2` | (144,) |
| Output | logits | (144 raw scores) | (144,) |

### What Are These Parameters?

```python
self.theta_W1 = nn.Parameter(torch.randn(hidden, 36) * 0.01)  # 128×36 = 4,608 numbers
self.theta_b1 = nn.Parameter(torch.zeros(hidden))              # 128 numbers
self.theta_W2 = nn.Parameter(torch.randn(144, hidden) * 0.01)  # 144×128 = 18,432 numbers
self.theta_b2 = nn.Parameter(torch.zeros(144))                 # 144 numbers
```

Total θ parameters: ~23,312 numbers

- **W1** transforms the 36-number board into 128 intermediate numbers
- **W2** transforms those 128 numbers into 144 output logits
- **b1** and **b2** are biases (shifts added after each layer)

### Why `nn.Parameter`?

```python
self.theta_W1 = nn.Parameter(torch.randn(hidden, 36) * 0.01)
```

`nn.Parameter` tells PyTorch: "this tensor should be optimized (gradients should flow through it)."

Without `nn.Parameter`, PyTorch's Adam optimizer would refuse to update the tensor.

### Why Multiply by 0.01?

Random initialization with very large weights causes the network to output extreme values early on, making training chaotic. Initializing small (0.01) means the network starts with "cautious" predictions close to zero.

---

### What Are Logits?

Logits are raw scores — any number. Not probabilities yet.

Example:
```
logits = [0.5, -0.3, 2.1, 1.8, -0.9, 0.2, ..., 3.2]
```

Each number corresponds to one of the 144 possible actions:
- logits[0] = score for action 0
- logits[47] = score for action 47
- logits[143] = score for action 143

Positive logit = "I think this is a good action"
Negative logit = "I think this is a bad action"

Later, softmax converts these to probabilities.

---

## Part 3: The Value Network Works the Same Way (w)

```python
def _get_value(self, obs_t):
    hidden = torch.relu(obs_t @ self.w_W1.T + self.w_b1)
    return (hidden @ self.w_W2.T + self.w_b2).squeeze(-1)
```

Identical structure, but:
- Input: 36 numbers (board state)
- Hidden: 128 numbers
- Output: 1 number (the value estimate)

The single output number answers: "How good is this board position?" on a scale the network learns.

---

## Part 4: Understanding `select_action`

```python
def select_action(self, obs, action_mask):
    obs_t = torch.FloatTensor(obs)
    mask_t = torch.FloatTensor(action_mask)

    with torch.no_grad():
        logits = self._get_policy_logits(obs_t)
        masked_logits = logits + (mask_t - 1.0) * 1e9
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
    return action.item()
```

### Step 1: Convert to Tensors

```python
obs_t = torch.FloatTensor(obs)        # board: (36,)
mask_t = torch.FloatTensor(action_mask)  # legal actions: (144,)
```

### Step 2: Get Logits

```python
logits = self._get_policy_logits(obs_t)  # (144,)
```

Example output:
```
logits = [0.5, -0.3, 2.1, 1.8, -0.9, 0.2, ..., 3.2]
```

### Step 3: Action Masking (The Trick)

```python
masked_logits = logits + (mask_t - 1.0) * 1e9
```

This line prevents illegal actions from being picked. Let's see how:

**For legal actions** (mask = 1):
```
masked_logits[legal] = logits[legal] + (1 - 1.0) * 1e9
                     = logits[legal] + 0
                     = logits[legal]  ← unchanged
```

**For illegal actions** (mask = 0):
```
masked_logits[illegal] = logits[illegal] + (0 - 1.0) * 1e9
                       = logits[illegal] - 1,000,000,000
                       = approximately -∞
```

Result:
- Legal actions keep their logits (maybe 0.5, 2.1, etc.)
- Illegal actions become huge negative numbers (-1e9)

### Step 4: Convert Logits to Probabilities

```python
dist = Categorical(logits=masked_logits)
```

This applies softmax:

```
P(action a) = exp(masked_logits[a]) / (sum of exp(masked_logits[all]))
```

Example:
```
exp(-1e9) ≈ 0        (illegal action: zero probability)
exp(2.1) ≈ 8.2       (legal action with score 2.1)
```

When softmax divides, illegal actions (with probability 0) get zero, and legal actions are weighted by their logits.

### Step 5: Sample

```python
action = dist.sample()  # randomly pick one action based on probabilities
return action.item()    # convert tensor to plain Python integer
```

**Result:** The agent cannot pick an illegal move — illegal actions have zero probability.

### Why Mask Before Softmax?

You might think: "apply softmax first, then zero out illegal actions, then renormalize."

```python
# Bad way (don't do this)
probs = softmax(logits)
probs = probs * mask_t
probs = probs / sum(probs)
```

This has numerical issues:
- If many actions are illegal, `sum(probs)` becomes tiny
- Dividing by a tiny number causes floating-point errors

Masking before softmax is cleaner and numerically stable — just add a huge negative number. The softmax naturally handles it.

### Why `torch.no_grad()`?

```python
with torch.no_grad():
    logits = self._get_policy_logits(obs_t)
    ...
```

We're only **sampling** an action, not training. `torch.no_grad()` tells PyTorch: "don't build a computation graph for this." This saves memory since we won't compute gradients.

---

## Part 5: Understanding `update` — Training Step

```python
def update(self, obs, action_mask, action, reward, next_obs, done, I):
    obs_t = torch.FloatTensor(obs)

    # Compute value estimates
    v_s = self._get_value(obs_t)                   # WITH gradients

    if done:
        v_s_next = 0.0
    else:
        with torch.no_grad():
            v_s_next = self._get_value(torch.FloatTensor(next_obs)).item()

    # Compute TD error
    delta = reward + self.gamma * v_s_next - v_s.item()

    # === CRITIC UPDATE ===
    value_loss = -delta * v_s
    self.w_optimizer.zero_grad()
    value_loss.backward()                          # Compute gradients for w
    self.w_optimizer.step()                        # Update w

    # === ACTOR UPDATE ===
    logits = self._get_policy_logits(obs_t)        # WITH gradients
    mask_t = torch.FloatTensor(action_mask)
    masked_logits = logits + (mask_t - 1.0) * 1e9
    dist = Categorical(logits=masked_logits)
    log_prob = dist.log_prob(torch.tensor(action))

    policy_loss = -I * delta * log_prob
    self.theta_optimizer.zero_grad()
    policy_loss.backward()                         # Compute gradients for theta
    self.theta_optimizer.step()                    # Update theta

    return delta
```

This method has two forward passes and two backward passes (two training steps).

---

## Part 6: What Is `.backward()` Really Doing?

This is the most important part to understand.

### Computation Graphs

When you write:

```python
v_s = self._get_value(obs_t)
```

PyTorch doesn't just compute a number — it builds a **computation graph** that records every operation:

```
obs_t (36 numbers)
    ↓ matmul
w_W1.T (128×36)
    ↓ add
w_b1 (128 numbers)
    ↓ relu
hidden (128 numbers, some zeros)
    ↓ matmul
w_W2.T (128×1)
    ↓ add
w_b2 (1 number)
    ↓
v_s (1 number: the value estimate)
```

Every operation is recorded. PyTorch knows which parameters (w_W1, w_b1, w_W2, w_b2) participated in producing v_s.

### Backpropagation: Tracing Backwards

When you call:

```python
value_loss = -delta * v_s
value_loss.backward()
```

PyTorch traces the graph **backwards** using the chain rule:

```
value_loss = -delta * v_s
    ↓
∂(value_loss)/∂v_s = -delta

v_s = ... (some computation)
    ↓
∂(value_loss)/∂w_W2 = ∂(value_loss)/∂v_s × ∂v_s/∂w_W2 (chain rule)
∂(value_loss)/∂w_W1 = ∂(value_loss)/∂v_s × ∂v_s/∂w_W1
... (for all w parameters)
```

### A Concrete Example

Let's trace a tiny network:

```python
W = nn.Parameter(torch.tensor([3.0]))
b = nn.Parameter(torch.tensor([0.0]))
obs = torch.tensor([2.0])

# Forward pass
output = obs * W + b     # = 2*3 + 0 = 6

# Loss
delta = 1.0
loss = -delta * output   # = -1 * 6 = -6

# Backward pass
loss.backward()
```

PyTorch computes:

```
∂loss/∂output = -delta = -1

∂output/∂W = obs = 2    (because output depends on W through multiplication by obs)

∂loss/∂W = ∂loss/∂output × ∂output/∂W
         = (-1) × 2
         = -2

∂output/∂b = 1          (because output = ... + b)

∂loss/∂b = ∂loss/∂output × ∂output/∂b
         = (-1) × 1
         = -1
```

After `.backward()`:
```python
print(W.grad)  # -2.0
print(b.grad)  # -1.0
```

These gradients tell you:
- If you increase W, the loss **decreases** (gradient is negative)
- If you increase b, the loss **decreases** (gradient is negative)

### How the Optimizer Uses Gradients

```python
self.w_optimizer.step()
```

This applies **gradient descent**:

```
w_new = w - learning_rate × ∂loss/∂w
```

From our example:
```
W_new = W - 0.01 × (-2.0)
      = 3.0 - (-0.02)
      = 3.02   ← W increased because we wanted to decrease loss

b_new = b - 0.01 × (-1.0)
      = 0.0 - (-0.01)
      = 0.01   ← b increased
```

Both parameters moved in the direction that **reduces** the loss.

---

### Why Two `.backward()` Calls?

In `update()`, you have:

```python
# Backward pass #1: for the critic
value_loss.backward()
self.w_optimizer.step()

# Backward pass #2: for the actor
policy_loss.backward()
self.theta_optimizer.step()
```

These are **independent graphs**:
- `value_loss = -delta * v_s` depends only on w parameters
- `policy_loss = -I * delta * log_prob` depends only on theta parameters

You trace back through the value network, update w, then trace back through the policy network, update theta.

---

### Why `.zero_grad()` Before `.backward()`?

```python
self.w_optimizer.zero_grad()
value_loss.backward()
```

`.zero_grad()` clears old gradients. Without it, new gradients would **accumulate** on top of old ones (add to them). We want fresh gradients, so we clear first.

---

## Part 7: The Full Training Loop

Here's how everything fits together:

```
STEP 1: Collect Experience
  select_action(board)
      └─ forward pass (no grad)
      └─ mask illegal actions
      └─ sample action
  env.step(action) → reward, next_board

STEP 2: Train on Experience
  update(board, action, reward, next_board)
      ├─ forward pass #1: v_s = network(board)    [WITH grad]
      ├─ forward pass #2: logits = network(board) [WITH grad]
      ├─ compute value_loss
      ├─ value_loss.backward()  ← compute ∂loss/∂w
      ├─ w_optimizer.step()     ← w ← w - lr × ∂loss/∂w
      ├─ compute policy_loss
      ├─ policy_loss.backward()  ← compute ∂loss/∂theta
      └─ theta_optimizer.step()  ← theta ← theta - lr × ∂loss/∂theta

STEP 3: Repeat
  select_action(next_board) with updated weights
      └─ network produces different logits because weights changed
  ... and so on
```

Every step, the weights change slightly. After thousands of steps, the network has learned to output good logits for good actions.

---

## Part 8: Why This Works

### The Magic of Automatic Differentiation

For our 24,000-parameter network, computing gradients by hand would be pages of calculus. PyTorch computes them automatically using `.backward()`.

That's the power of automatic differentiation — it makes training neural networks **tractable**.

### The Magic of Gradient Descent

After each step, weights move slightly in the direction that reduces loss:

```
iteration 0:  weights = random
iteration 1:  weights move toward better values
iteration 2:  weights move more
...
iteration 30000:  weights have converged to good values
```

The network learns to:
- Output high logits for actions that lead to good outcomes
- Output low logits for actions that lead to bad outcomes
- Estimate the value of board positions accurately

All through gradient descent and backpropagation.

---

## Part 9: Key Hyperparameters and Why They Matter

| Parameter | Value | Role |
|-----------|-------|------|
| α^θ (policy lr) | 1e-4 | Step size for weight updates in policy network. Smaller = more cautious learning |
| α^w (value lr) | 1e-3 | Step size for weight updates in value network. Larger than θ because critic needs to catch up |
| Hidden size | 128 | Number of intermediate neurons. Larger = more capacity but slower training |
| Initial weight scale | 0.01 | Weights start this small. Prevents extreme early predictions |
| 1e9 (masking constant) | 1e9 | Large enough that exp(-1e9) ≈ 0, so illegal actions get zero probability |

---

## Summary

1. **Neural networks** are a way to parameterize the policy π and value function v̂
2. **Forward pass** (in `select_action`): board → network → logits → probabilities → sample action
3. **Masking** (in `select_action`): add -1e9 to illegal logits before softmax so they get zero probability
4. **Loss computation** (in `update`): value_loss and policy_loss combine network outputs with the TD error
5. **Backward pass** (in `update`): `.backward()` computes gradients using the chain rule through the entire computation graph
6. **Optimizer step**: weights move in the direction that reduces loss
7. **Repeat**: after thousands of iterations, the network learns good policies and value estimates

This entire process is **end-to-end differentiable** — every piece can have gradients flow through it.
