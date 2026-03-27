import random
import numpy as np
import torch
from mycheckersenv import CheckersEnv
from myagent import CheckersAgent

# Hyperparameters
PRETRAIN_EPISODES = 15000       # Phase 1: train vs random
SELFPLAY_EPISODES = 15000       # Phase 2: self-play
MAX_STEPS_PER_EPISODE = 200

# Clone every SNAPSHOT_INTERVAL episodes for self-play pool
SNAPSHOT_INTERVAL = 100

MAX_POOL_SIZE = 10
EVAL_INTERVAL = 2500
EVAL_GAMES = 50
REWARD_SHAPING_SCALE = 0.2


def _get_material(board, agent_name):
    """Get material score from agent's perspective."""
    p0_pieces = np.sum(board == 1)
    p0_kings = np.sum(board == 2)
    p1_pieces = np.sum(board == -1)
    p1_kings = np.sum(board == -2)
    if agent_name == "player_0":
        return (p0_pieces + 2.0 * p0_kings) - (p1_pieces + 2.0 * p1_kings)
    else:
        return (p1_pieces + 2.0 * p1_kings) - (p0_pieces + 2.0 * p0_kings)


def random_opponent(obs, action_mask):
    """Random opponent that picks a legal action uniformly."""
    legal = np.where(action_mask == 1)[0]
    return np.random.choice(legal)


def make_agent_opponent(opponent_agent):
    """Create opponent function from a CheckersAgent."""
    def opponent_fn(obs, action_mask):
        return opponent_agent.select_action(obs, action_mask)
    return opponent_fn


def play_episode(env, agent, opponent_fn):
    """Play one episode with per-step Actor-Critic updates.

    Follows the textbook one-step AC algorithm:
        - At each agent step: observe S, take A, receive R, observe S'
        - Compute TD error delta and update policy + value networks
        - Decay discount accumulator I <- gamma * I

    The opponent's turns are part of the environment transition —
    from the agent's view, S' is the board state at its next turn.
    """
    env.reset()

    # Randomly assign the learning agent to a seat
    agent_seat = random.choice(["player_0", "player_1"])

    # One-step AC state: I starts at 1 per episode (textbook)
    I = 1.0
    prev_obs = None
    prev_mask = None
    prev_action = None
    prev_material = None

    step_count = 0
    winner = None

    for current_agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            if current_agent == agent_seat and prev_obs is not None:
                # Terminal transition: do final update with v_hat(S') = 0
                material_now = _get_material(env.board, agent_seat)
                shaped_r = REWARD_SHAPING_SCALE * (material_now - prev_material)
                R = reward + shaped_r
                agent.update(prev_obs, prev_mask, prev_action, R, None, True, I)

                winner = "agent" if reward > 0 else "opponent" if reward < 0 else None

            env.step(None)
            continue

        board_obs = obs["observation"]
        action_mask = obs["action_mask"]

        if current_agent == agent_seat:
            # Agent's turn — first handle the pending update from last step
            if prev_obs is not None:
                # We now have S' (current obs). Complete the update for the
                # previous transition: (prev_obs, prev_action) -> (current obs)
                material_now = _get_material(env.board, agent_seat)
                shaped_r = REWARD_SHAPING_SCALE * (material_now - prev_material)
                agent.update(prev_obs, prev_mask, prev_action, shaped_r,
                             board_obs, False, I)
                I *= agent.gamma  # I <- gamma * I

            # Select action A ~ pi(.|S, theta)
            action = agent.select_action(board_obs, action_mask)

            # Store transition state for next update
            prev_obs = board_obs.copy()
            prev_mask = action_mask.copy()
            prev_action = action
            prev_material = _get_material(env.board, agent_seat)
        else:
            # Opponent's turn — just pick and step, no recording
            action = opponent_fn(board_obs, action_mask)

        env.step(action)
        step_count += 1

        # Max step truncation (draw)
        if step_count >= MAX_STEPS_PER_EPISODE:
            if prev_obs is not None:
                material_now = _get_material(env.board, agent_seat)
                shaped_r = REWARD_SHAPING_SCALE * (material_now - prev_material)
                agent.update(prev_obs, prev_mask, prev_action, shaped_r,
                             None, True, I)
            break

    return winner


def evaluate_vs_random(agent, num_games=EVAL_GAMES):
    """Play games against random and return win rate."""
    wins = 0
    for _ in range(num_games):
        env = CheckersEnv()
        env.reset()
        agent_seat = random.choice(["player_0", "player_1"])
        step_count = 0

        for current_agent in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            if terminated or truncated:
                if reward > 0 and current_agent == agent_seat:
                    wins += 1
                env.step(None)
                continue

            board_obs = obs["observation"]
            action_mask = obs["action_mask"]

            if current_agent == agent_seat:
                action = agent.select_action(board_obs, action_mask)
            else:
                legal = np.where(action_mask == 1)[0]
                action = np.random.choice(legal)

            env.step(action)
            step_count += 1
            if step_count >= MAX_STEPS_PER_EPISODE:
                break

    return wins / num_games


def select_opponent(opponent_pool):
    """80% latest snapshot, 20% random from pool."""
    if random.random() < 0.8:
        return opponent_pool[-1]
    return random.choice(opponent_pool)


def train():
    """Main training loop: pre-train vs random, then self-play."""
    agent = CheckersAgent(alpha_theta=1e-4, alpha_w=1e-3, gamma=0.99)
    env = CheckersEnv()

    total_episodes = PRETRAIN_EPISODES + SELFPLAY_EPISODES
    opponent_pool = []

    total_wins = 0
    total_games = 0

    print("=" * 60)
    print("Phase 1: Pre-training vs Random Opponent")
    print("=" * 60)

    for episode in range(1, total_episodes + 1):
        # Switch phases
        if episode == PRETRAIN_EPISODES + 1:
            print("\n" + "=" * 60)
            print("Phase 2: Self-Play Training")
            print("=" * 60)
            # Lower policy lr for self-play (finer tuning)
            for pg in agent.theta_optimizer.param_groups:
                pg["lr"] = 5e-5
            opponent_pool = [agent.clone()]

        # Select opponent
        if episode <= PRETRAIN_EPISODES:
            opponent_fn = random_opponent
        else:
            # 30% random to prevent forgetting, 70% self-play
            if random.random() < 0.3:
                opponent_fn = random_opponent
            else:
                opp = select_opponent(opponent_pool)
                opponent_fn = make_agent_opponent(opp)

        winner = play_episode(env, agent, opponent_fn)

        if winner == "agent":
            total_wins += 1
        total_games += 1

        # Snapshot for self-play
        if episode > PRETRAIN_EPISODES and episode % SNAPSHOT_INTERVAL == 0:
            opponent_pool.append(agent.clone())
            if len(opponent_pool) > MAX_POOL_SIZE:
                opponent_pool.pop(0)

        # Evaluate and log
        if episode % EVAL_INTERVAL == 0:
            wr = evaluate_vs_random(agent)

            phase = "pretrain" if episode <= PRETRAIN_EPISODES else "selfplay"
            sp_wr = total_wins / max(total_games, 1)
            print(f"[{phase}] Ep {episode:>6} | "
                  f"vs Random: {wr:.1%} | "
                  f"Train WR: {sp_wr:.1%} | "
                  f"Pool: {len(opponent_pool)}")

            total_wins = 0
            total_games = 0

    # Save
    agent.save("trained_agent.pth")
    print("\nTraining complete. Model saved to trained_agent.pth")

    final_wr = evaluate_vs_random(agent, num_games=100)
    print(f"Final win rate vs random (100 games): {final_wr:.1%}")


if __name__ == "__main__":
    train()
