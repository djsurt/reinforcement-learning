"""
Sample run: trained Actor-Critic agent vs itself (self-play).
Shows board states step-by-step and prints final cumulative reward.
"""
import numpy as np
from mycheckersenv import CheckersEnv
from myagent import CheckersAgent


def render_board(board, turn):
    """Pretty-print the board with grid lines."""
    symbols = {0: ' . ', 1: ' o ', 2: ' O ', -1: ' x ', -2: ' X '}
    sep = "  +" + "---+" * 6
    col_header = "    " + "   ".join(str(c) for c in range(6))
    print(col_header)
    print(sep)
    for row in range(6):
        row_str = f"{row} |"
        for col in range(6):
            row_str += symbols[board[row][col]] + "|"
        print(row_str)
        print(sep)
    print(f"  Turn: {turn}\n")


def sample_run(model_path="trained_agent.pth", max_steps=200):
    # Load trained agent for both sides
    agent_p0 = CheckersAgent()
    agent_p0.load(model_path)

    agent_p1 = CheckersAgent()
    agent_p1.load(model_path)

    print(f"Loaded trained agent from {model_path} for both players\n")

    env = CheckersEnv(render_mode="human")
    env.reset()

    agents = {"player_0": agent_p0, "player_1": agent_p1}

    print("=" * 50)
    print("Sample Game: Trained Agent (o/O) vs Trained Agent (x/X)")
    print("         Self-Play Demonstration")
    print("=" * 50)
    print("\nSymbols:  o = player_0 piece   O = player_0 king")
    print("          x = player_1 piece   X = player_1 king")
    print("          . = empty\n")
    print("--- Initial Board ---")
    render_board(env.board, env.agent_selection)

    step_count = 0
    cumulative_rewards = {"player_0": 0.0, "player_1": 0.0}

    for current_agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()

        cumulative_rewards[current_agent] += reward

        if terminated or truncated:
            env.step(None)
            continue

        board_obs = obs["observation"]
        action_mask = obs["action_mask"]

        action = agents[current_agent].select_action(board_obs, action_mask)

        from_pos, to_pos = env.action_map[action]
        move_type = "jumps" if abs(to_pos[0] - from_pos[0]) == 2 else "moves"

        env.step(action)
        step_count += 1

        print(f"Step {step_count:>3} | {current_agent} {move_type} {from_pos} -> {to_pos}")
        render_board(env.board, env.agent_selection)

        if step_count >= max_steps:
            print(f"\nReached max steps ({max_steps}) — declaring a draw.")
            break

    # Final result
    p0_pieces = np.sum(env.board > 0)
    p1_pieces = np.sum(env.board < 0)

    print("\n" + "=" * 50)
    print("GAME RESULT")
    print("=" * 50)
    print(f"Total steps: {step_count}")
    print(f"player_0 pieces remaining: {p0_pieces}")
    print(f"player_1 pieces remaining: {p1_pieces}")

    if p1_pieces == 0:
        print("Winner: player_0")
    elif p0_pieces == 0:
        print("Winner: player_1")
    else:
        print("Result: Draw / Max steps reached")

    print(f"\nFinal Cumulative Reward — player_0: {cumulative_rewards['player_0']}")
    print(f"Final Cumulative Reward — player_1: {cumulative_rewards['player_1']}")
    print("=" * 50)


if __name__ == "__main__":
    sample_run()
