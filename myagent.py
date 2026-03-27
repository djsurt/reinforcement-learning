import copy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class CheckersAgent:
    """One-step Actor-Critic agent (Sutton & Barto, episodic).

    Function approximation using PyTorch for automatic differentiation:
        - pi(a|s, theta): differentiable policy parameterization
        - v_hat(s, w):    differentiable state-value function parameterization

    theta and w each use a hidden layer (36 -> 128 -> output) for
    expressiveness. PyTorch autograd computes the gradients; the actual
    parameter updates follow the textbook equations exactly:

        delta = R + gamma * v_hat(S', w) - v_hat(S, w)
        w     <- w     + alpha_w     * delta * grad v_hat(S, w)
        theta <- theta + alpha_theta * I * delta * grad ln pi(A|S, theta)
        I     <- gamma * I
    """

    def __init__(self, alpha_theta=1e-4, alpha_w=1e-3, gamma=0.99):
        self.gamma = gamma

        hidden = 128

        # ---- Policy parameters theta ----
        # theta maps state (36) -> hidden (128) -> action logits (144)
        self.theta_W1 = nn.Parameter(torch.randn(hidden, 36) * 0.01)
        self.theta_b1 = nn.Parameter(torch.zeros(hidden))
        self.theta_W2 = nn.Parameter(torch.randn(144, hidden) * 0.01)
        self.theta_b2 = nn.Parameter(torch.zeros(144))

        # ---- Value weights w ----
        # w maps state (36) -> hidden (128) -> scalar value (1)
        self.w_W1 = nn.Parameter(torch.randn(hidden, 36) * 0.01)
        self.w_b1 = nn.Parameter(torch.zeros(hidden))
        self.w_W2 = nn.Parameter(torch.randn(1, hidden) * 0.01)
        self.w_b2 = nn.Parameter(torch.zeros(1))

        # Adam optimizers for adaptive step sizes (alpha_theta, alpha_w)
        self.theta_optimizer = torch.optim.Adam(
            [self.theta_W1, self.theta_b1, self.theta_W2, self.theta_b2],
            lr=alpha_theta
        )
        self.w_optimizer = torch.optim.Adam(
            [self.w_W1, self.w_b1, self.w_W2, self.w_b2],
            lr=alpha_w
        )

    @property
    def _theta_params(self):
        """All policy parameters (theta)."""
        return [self.theta_W1, self.theta_b1, self.theta_W2, self.theta_b2]

    @property
    def _w_params(self):
        """All value parameters (w)."""
        return [self.w_W1, self.w_b1, self.w_W2, self.w_b2]

    # Forward pass through policy network to get pi(.|s, theta) logits
    def _get_policy_logits(self, obs_t):
        """pi(.|s, theta): compute action logits from state."""
        hidden = torch.relu(obs_t @ self.theta_W1.T + self.theta_b1)
        return hidden @ self.theta_W2.T + self.theta_b2

    # Forward pass through value network to get v_hat(s, w)
    def _get_value(self, obs_t):
        """v_hat(s, w): compute state value from state."""
        hidden = torch.relu(obs_t @ self.w_W1.T + self.w_b1)
        return (hidden @ self.w_W2.T + self.w_b2).squeeze(-1)

    def select_action(self, obs, action_mask):
        """Sample action A ~ pi(.|S, theta) with action masking."""
        obs_t = torch.FloatTensor(obs)
        mask_t = torch.FloatTensor(action_mask)

        with torch.no_grad():
            logits = self._get_policy_logits(obs_t)

            # Masking illegal actions: set their logits to a very large negative number
            masked_logits = logits + (mask_t - 1.0) * 1e9

            #Soft max applied to logits. Masked illegal actions have near-zero prob.
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
        return action.item()

    def update(self, obs, action_mask, action, reward, next_obs, done, I):
        """One-step Actor-Critic update (called after every agent step).

        Implements the textbook update rules:
            delta = R + gamma * v_hat(S', w) - v_hat(S, w)
            w     <- w     + alpha_w * delta * grad v_hat(S, w)
            theta <- theta + alpha_theta * I * delta * grad ln pi(A|S, theta)

        Returns:
            delta (float): the TD error
        """
        obs_t = torch.FloatTensor(obs)

        # ---- delta = R + gamma * v_hat(S', w) - v_hat(S, w) ----
        v_s = self._get_value(obs_t)

        if done:
            v_s_next = 0.0  # if S' is terminal, v_hat(S', w) = 0
        else:
            with torch.no_grad():
                v_s_next = self._get_value(torch.FloatTensor(next_obs)).item()

        delta = reward + self.gamma * v_s_next - v_s.item()

        # ---- w <- w + alpha_w * delta * grad v_hat(S, w) ----
        # Loss = -delta * v_hat(S, w), so gradient descent gives the textbook update
        # Sign flip because we want to ascend the policy gradient, but optimizers do descent
        value_loss = -delta * v_s
        self.w_optimizer.zero_grad()
        value_loss.backward()
        self.w_optimizer.step()

        # ---- theta <- theta + alpha_theta * I * delta * grad ln pi(A|S, theta) ----
        logits = self._get_policy_logits(obs_t)
        mask_t = torch.FloatTensor(action_mask)

        # This is done so that the illegal actions are very negative.
        masked_logits = logits + (mask_t - 1.0) * 1e9

        # Soft max applied to logits.
        dist = Categorical(logits=masked_logits)
        log_prob = dist.log_prob(torch.tensor(action))

        # Sign flip because we want to ascend the policy gradient, but optimizers do descent
        policy_loss = -I * delta * log_prob
        self.theta_optimizer.zero_grad()
        policy_loss.backward()
        self.theta_optimizer.step()

        return delta

    def clone(self):
        """Create a copy for opponent snapshots."""
        new_agent = CheckersAgent.__new__(CheckersAgent)
        new_agent.gamma = self.gamma
        new_agent.theta_optimizer = None
        new_agent.w_optimizer = None
        # Deep copy all parameters (no grad needed for opponent)
        for attr in ["theta_W1", "theta_b1", "theta_W2", "theta_b2",
                      "w_W1", "w_b1", "w_W2", "w_b2"]:
            data = getattr(self, attr).data.clone()
            setattr(new_agent, attr, data)
        return new_agent

    def save(self, path):
        torch.save({
            attr: getattr(self, attr).data
            for attr in ["theta_W1", "theta_b1", "theta_W2", "theta_b2",
                          "w_W1", "w_b1", "w_W2", "w_b2"]
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, weights_only=True)
        for attr in checkpoint:
            setattr(self, attr, checkpoint[attr].requires_grad_(True))
