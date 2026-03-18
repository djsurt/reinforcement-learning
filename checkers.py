import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
import gymnasium.spaces as spaces

N_ACTIONS = 144  # 18 dark squares * 8 moves (4 single-step + 4 two-step)

class CheckersEnv(AECEnv):
    metadata = {"render_modes": ["human", "ansi"], "name": "checkers_v0"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]
        self.action_map = self._build_action_map()
        # Need to fill this in
        self.observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(low=-2, high=2, shape=(36,), dtype=np.int8),
                "action_mask": spaces.Box(low=0, high=1, shape=(N_ACTIONS, ), dtype=np.int8),
            }) 
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Discrete(N_ACTIONS)
            for agent in self.possible_agents
        }

        self.render_mode = render_mode

    def _build_action_map(self):
        action_map = {}
        action_id = 0
        # Encoding action_id as (from_row, from_col) -> (to_row, to_col)
        for row in range(6):
            for col in range(6):
                if (row + col) % 2 == 1:  # Only dark squares are valid
                    # 1-step diagonal moves
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        action_map[action_id] = ((row, col), (row + dr, col + dc))
                        action_id += 1
                    # 2-step diagonal moves (jumps)
                    for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                        action_map[action_id] = ((row, col), (row + dr, col + dc))
                        action_id += 1
        return action_map
    
    def _init_board(self):
        board = np.zeros((6, 6), dtype=np.int8)
        for row in range(6):
            for col in range(6):
                if (row + col) % 2 == 1:
                    if row < 2:
                        board[row][col] = -1 # player_1's pieces
                    elif row > 3:
                        board[row][col] = 1 # player_0's pieces
        return board

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns an initial observation and info dict."""
        self.agents = self.possible_agents[:]
        self.board = self._init_board()

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        return self._observe(self.agent_selection), self.infos[self.agent_selection]
    
    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of episode is reached, you are responsible for calling `reset()` to reset this environment's state."""
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        
        self._cumulative_rewards[agent] = 0
        self._apply_action(agent, action)

        winner = self._check_winner()
        if winner is not None:
            for a in self.agents:
                self.terminations[a] = True
                self.rewards[a] = 1 if a == winner else -1
        
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()
    
    def observe(self, agent):
        return self._observe(agent)

    def _observe(self, agent):
        obs = self.board.flatten().copy()
        if agent == "player_1":
            obs = -obs #flip perspective so your pieces are always positive
        mask = self._get_action_mask(agent)
        return {"observation": obs, "action_mask": mask}

    def render(self):
        if self.render_mode in ("human", "ansi"):
            symbols = {0: '.', 1: 'o', 2: 'O', -1: 'x', -2: 'X'}
            print(" " + " ".join(str(c) for c in range(6)))
            for row in range(6):
                row_str = str(row) + " "
                row_str += " ".join(symbols[self.board[row][col]] for col in range(6))
                print(row_str)
            print(f" Turn: {self.agent_selection}")
            print()
    
    def close(self):
        pass

    def _get_direction(self, piece, is_king):
        if is_king:
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        # Player 0, pieces move up (-1), Player 1 pieces move down (+1)
        if piece > 0:
            return [(-1, -1), (-1, 1)]
        else:
            return [(1, -1), (1, 1)]
    
    def _get_legal_moves(self, agent):
        player_val = 1 if agent == "player_0" else -1
        jumps = []
        moves = []
        for row in range(6):
            for col in range(6):
                piece = self.board[row][col]
                if piece * player_val <= 0: # not your piece
                    continue
                is_king = abs(piece) == 2
                directions = self._get_direction(piece, is_king)

                for dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    # Simple move
                    if 0 <= nr < 6 and 0 <= nc < 6 and self.board[nr][nc] == 0:
                        moves.append(((row, col), (nr, nc)))
                    # Jump
                    jr, jc = row + 2 * dr, col + 2 * dc
                    mid_r, mid_c = row + dr, col + dc
                    if (0 <= jr < 6 and 0 <= jc < 6 
                        and self.board[jr][jc] == 0
                        and 0 <= mid_r < 6 and 0 <= mid_c < 6
                        and self.board[mid_r][mid_c] * player_val < 0):
                        jumps.append(((row, col), (jr, jc)))
        # If jumps are available, you must take them
        return jumps if jumps else moves

    def _get_action_mask(self, agent):
        mask = np.zeros(N_ACTIONS, dtype=np.int8)
        legal_moves = self._get_legal_moves(agent)
        legal_set = set(legal_moves)
        for action_id, (from_pos, to_pos) in self.action_map.items():
            if (from_pos, to_pos) in legal_set:
                mask[action_id] = 1
        return mask
    
    def _apply_action(self, agent, action):
        from_pos, to_pos = self.action_map[action]
        fr, fc = from_pos
        tr, tc = to_pos

        self.board[tr][tc] = self.board[fr][fc]
        self.board[fr][fc] = 0

        #If jump, remove captured piece
        if abs(tr - fr) == 2:
            mid_r = (fr + tr) // 2
            mid_c = (fc + tc) // 2
            self.board[mid_r][mid_c] = 0
        
        # King promtion
        player_val = 1 if agent == "player_0" else -1
        if player_val == 1 and tr == 0:
            self.board[tr][tc] = 2
        if player_val == -1 and tr == 5:
            self.board[tr][tc] = -2
    
    def _check_winner(self):
        p0_pieces = np.sum(self.board > 0)
        p1_pieces = np.sum(self.board < 0)

        if p1_pieces == 0:
            return "player_0"
        if p0_pieces == 0:
            return "player_1"

        current = self.agent_selection
        if len(self._get_legal_moves(current)) == 0:
            return "player_1" if current == "player_0" else "player_0"
        return None