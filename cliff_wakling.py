# Import necessary libraries
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv
from gymnasium.error import DependencyNotInstalled
from os import path

# Do not change this class
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
image_path = path.join(path.dirname(gym.__file__), "envs", "toy_text")


class CliffWalking(CliffWalkingEnv):
    def __init__(self, is_hardmode=True, num_cliffs=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_hardmode = is_hardmode

        # Generate random cliff positions
        if self.is_hardmode:
            self.num_cliffs = num_cliffs
            self._cliff = np.zeros(self.shape, dtype=bool)
            self.start_state = (3, 0)
            self.terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
            self.cliff_positions = []
            while len(self.cliff_positions) < self.num_cliffs:
                new_row = np.random.randint(0, 4)
                new_col = np.random.randint(0, 11)
                state = (new_row, new_col)
                if (
                        (state not in self.cliff_positions)
                        and (state != self.start_state)
                        and (state != self.terminal_state)
                ):
                    self._cliff[new_row, new_col] = True
                    if not self.is_valid():
                        self._cliff[new_row, new_col] = False
                        continue
                    self.cliff_positions.append(state)

        # Calculate transition probabilities and rewards
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

    def _calculate_transition_prob(self, current, delta):
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, self.start_state_index, -100, False)]

        terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        is_terminated = tuple(new_position) == terminal_state
        return [(1 / 3, new_state, -1, is_terminated)]

    # DFS to check that it's a valid path.
    def is_valid(self):
        frontier, discovered = [], set()
        frontier.append((3, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= self.shape[0] or c_new < 0 or c_new >= self.shape[1]:
                        continue
                    if (r_new, c_new) == self.terminal_state:
                        return True
                    if not self._cliff[r_new][c_new]:
                        frontier.append((r_new, c_new))
        return False

    def step(self, action):
        if action not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid action {action}   must be in [0, 1, 2, 3]")

        if self.is_hardmode:
            match action:
                case 0:
                    action = np.random.choice([0, 1, 3], p=[1 / 3, 1 / 3, 1 / 3])
                case 1:
                    action = np.random.choice([0, 1, 2], p=[1 / 3, 1 / 3, 1 / 3])
                case 2:
                    action = np.random.choice([1, 2, 3], p=[1 / 3, 1 / 3, 1 / 3])
                case 3:
                    action = np.random.choice([0, 2, 3], p=[1 / 3, 1 / 3, 1 / 3])

        return super().step(action)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e
        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("CliffWalking - Edited by Audrina & Kian")
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.elf_images is None:
            hikers = [
                path.join(image_path, "img/elf_up.png"),
                path.join(image_path, "img/elf_right.png"),
                path.join(image_path, "img/elf_down.png"),
                path.join(image_path, "img/elf_left.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in hikers
            ]
        if self.start_img is None:
            file_name = path.join(image_path, "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(image_path, "img/cookie.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.mountain_bg_img is None:
            bg_imgs = [
                path.join(image_path, "img/mountain_bg1.png"),
                path.join(image_path, "img/mountain_bg2.png"),
            ]
            self.mountain_bg_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in bg_imgs
            ]
        if self.near_cliff_img is None:
            near_cliff_imgs = [
                path.join(image_path, "img/mountain_near-cliff1.png"),
                path.join(image_path, "img/mountain_near-cliff2.png"),
            ]
            self.near_cliff_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in near_cliff_imgs
            ]
        if self.cliff_img is None:
            file_name = path.join(image_path, "img/mountain_cliff.png")
            self.cliff_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        for s in range(self.nS):
            row, col = np.unravel_index(s, self.shape)
            pos = (col * self.cell_size[0], row * self.cell_size[1])
            check_board_mask = row % 2 ^ col % 2
            self.window_surface.blit(self.mountain_bg_img[check_board_mask], pos)

            if self._cliff[row, col]:
                self.window_surface.blit(self.cliff_img, pos)
            if s == self.start_state_index:
                self.window_surface.blit(self.start_img, pos)
            if s == self.nS - 1:
                self.window_surface.blit(self.goal_img, pos)
            if s == self.s:
                elf_pos = (pos[0], pos[1] - 0.1 * self.cell_size[1])
                last_action = self.lastaction if self.lastaction is not None else 2
                self.window_surface.blit(self.elf_images[last_action], elf_pos)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )


# defines reward for actions
def get_reward(current_position, goal_position, cliff_positions, act):
    goal_reward = 200
    cliff_fall_penalty = -100
    step_penalty = -1
    step_left_penalty = -2  # encouraging it to move forward

    if current_position == goal_position:
        return goal_reward
    # Check if the current position is a cliff
    elif current_position in cliff_positions:
        return cliff_fall_penalty
    elif act == 3:
        return step_left_penalty
    else:
        return step_penalty


# sets optimal values for each state
def value_iteration(env, gamma, itr_num):
    V = np.zeros(s_num)
    for i in range(itr_num):
        V_new = np.zeros(s_num)
        V_new[47] = 100
        for state in range(s_num - 1):
            Q_list = []
            for action in range(a_num):
                Q = 0
                transitions = env.P[state]
                if action == 0:
                    temp = transitions.pop(2)
                    key = 2
                elif action == 1:
                    temp = transitions.pop(3)
                    key = 3
                elif action == 2:
                    key = 0
                    temp = transitions.pop(0)
                else:
                    temp = transitions.pop(1)
                    key = 1
                # Bellman Equation
                for T in transitions.items():
                    trans_key = T[0]
                    T = T[1]
                    Q += T[0][0] * (get_reward(states_pos[state], (3, 11), env.cliff_positions, trans_key) + gamma * V[T[0][1]])

                Q_list.append(Q)
                transitions[key] = temp
            V_new[state] = max(Q_list)

        if np.max(np.abs(V_new - V)) < 1e-10:  # if converged ==> break
            break
        V = V_new

    return V


# Finds optimal policy using optimal values
def find_optimal_policy(env, optimal_V, gamma):
    optimal_policy = np.zeros(s_num, dtype=int)
    for state in range(s_num):
        Q_list = []
        for action in range(a_num):
            Q = 0
            transitions = env.P[state]
            if action == 0:
                temp = transitions.pop(2)
                key = 2
            elif action == 1:
                temp = transitions.pop(3)
                key = 3
            elif action == 2:
                key = 0
                temp = transitions.pop(0)
            else:
                temp = transitions.pop(1)
                key = 1
            # Bellman Equation
            for T in transitions.items():
                trans_key = T[0]
                T = T[1]
                Q += T[0][0] * (get_reward(states_pos[state], (3, 11), env.cliff_positions, trans_key) + gamma * optimal_V[T[0][1]])

            Q_list.append(Q)
            transitions[key] = temp

        optimal_policy[state] = np.argmax(Q_list)
    return optimal_policy


if __name__ == '__main__':

    # Create an environment
    env = CliffWalking(render_mode="human")
    observation, info = env.reset(seed=30)

    # converts state number to 2D position
    states_pos = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (0, 4), 5: (0, 5), 6: (0, 6), 7: (0, 7), 8: (0, 8),
                  9: (0, 9), 10: (0, 10), 11: (0, 11), 12: (1, 0), 13: (1, 1), 14: (1, 2), 15: (1, 3), 16: (1, 4), 17: (1, 5),
                  18: (1, 6), 19: (1, 7), 20: (1, 8), 21: (1, 9), 22: (1, 10), 23: (1, 11), 24: (2, 0), 25: (2, 1), 26: (2, 2),
                  27: (2, 3), 28: (2, 4), 29: (2, 5), 30: (2, 6), 31: (2, 7), 32: (2, 8), 33: (2, 9), 34: (2, 10), 35: (2, 11),
                  36: (3, 0), 37: (3, 1), 38: (3, 2), 39: (3, 3), 40: (3, 4), 41: (3, 5), 42: (3, 6), 43: (3, 7), 44: (3, 8),
                  45: (3, 9), 46: (3, 10), 47: (3, 11)}

    s_num = env.observation_space.n
    a_num = env.action_space.n
    gamma = 0.9
    V = np.zeros(s_num)
    iteration_num = 1000

    optimal_values = value_iteration(env, gamma, iteration_num)

    optimal_policy = find_optimal_policy(env, optimal_values, gamma)
    print("Optimal Policy:", optimal_policy)

    # Define the maximum number of iterations
    max_iter_number = 1000
    next_state = 36
    for __ in range(max_iter_number):

        action = optimal_policy[next_state]

        # Perform the action and receive feedback from the environment
        next_state, reward, done, truncated, info = env.step(action)

        if done:
            print("Cookie Found!")

        if done or truncated:
            observation, info = env.reset()

    # Close the environment
    env.close()
