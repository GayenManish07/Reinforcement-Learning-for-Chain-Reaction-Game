from copy import copy
from os import path
import numpy as np
import gymnasium
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
import pygame

class ChainReactionEnvironment(AECEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array",None],
        "name": "ChainReaction_v0",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(self, render_mode: str = None, screen_height: int = 800):
        super().__init__()


        self.agents = ["p1_team_a","p2_team_b","p3_team_a","p4_team_b"]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = None

        self.rewards = None
        self.infos = {name: {} for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.board = np.zeros((16, 16, 8), dtype=bool)
        self.board_history = np.zeros((16, 16, 32), dtype=bool)# set board history

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen_height = self.screen_width = screen_height
        self.screen = None

        if self.render_mode in ["human", "rgb_array"]:
            self.BOARD_SIZE = (self.screen_width, self.screen_height)
            self.clock = pygame.time.Clock()
            self.cell_size = (self.BOARD_SIZE[0] / 16, self.BOARD_SIZE[1] / 16)

            bg_name = path.join(path.dirname(__file__), "./images/grid.jpg")
            self.bg_image = pygame.transform.scale(
                pygame.image.load(bg_name), self.BOARD_SIZE
            )
            def load_piece(file_name):
                img_path = path.join(path.dirname(__file__), f"images/{file_name}.jpg")
                return pygame.transform.scale(
                    pygame.image.load(img_path), self.cell_size
                )

            self.piece_images = {
                "team_a": load_piece("team_a"),
                "team_b": load_piece("team_b"),
            }


    def observation_space(self, agent):

        self.observation_spaces = {
            name: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(16, 16, 40), dtype=bool
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(16*16,), dtype=np.int8
                    ),
                }
            )
            for name in self.agents
        }

        return self.observation_spaces[agent]
    

    def action_space(self, agent):

        self.action_spaces = {name: spaces.Discrete(16 * 16) for name in self.agents}

        return self.action_spaces[agent]
    

    def observe(self, agent):
        current_index = self.possible_agents.index(agent)

        observation = np.dstack(self.board,self.board_history)

        legal_moves = np.where(np.flatten(observation[:,:,(current_index+1)%2])==0) #board positions with no opponent particle(s)

        action_mask = np.zeros(16*16, "int8")
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}
    

    def reset(self, seed=None,):
        self.agents = self.possible_agents[:]

        self.board = np.zeros(shape=(16,16,8))
        self.num_steps = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.board_history = np.zeros(shape=(16, 16, 32), dtype=bool)

        if self.render_mode == "human":
            self.render()


    def step(self, action):# unfinished rewards and burst function
        
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        
        current_agent = self.agent_selection
        current_index = self.agents.index(current_agent)

        #convert action to coordinate on board (a//N, a%N)
        x_coord = action // 16
        y_coord = action % 16

        if self.board[x_coord, y_coord, (current_index+1)%2] == 1:          #check if opponent team has a particle in the position chosen by action 
            game_over = True
        elif self.board[x_coord, y_coord, (current_index)%2] == 0:          #check if friendly team has a particle in the position chosen by action
            self.board[x_coord, y_coord, (current_index)%2] = 1             #add particle if no particle is present 
            self.board[x_coord, y_coord, (current_index%2)*3 + 2] = 1       #change board state to track particle updates(0->1,1->2,2->3 or 3->0 with burst)
        elif self.board[x_coord, y_coord, (current_index%2)*3 + 2] == 1:
            self.board[x_coord, y_coord, (current_index%2)*3 + 2] = 0
            self.board[x_coord, y_coord, (current_index%2)*3 + 3] = 1
        elif self.board[x_coord, y_coord, (current_index%2)*3 + 3] == 1:
            self.board[x_coord, y_coord, (current_index%2)*3 + 3] = 0
            self.board[x_coord, y_coord, (current_index%2)*3 + 4] = 1
        elif self.board[x_coord, y_coord, (current_index%2)*3 + 4] == 1:
            self.board[x_coord, y_coord, (current_index%2)*3 + 4] = 0
            self.board[x_coord, y_coord, (current_index)%2] = 0
            self.burst(x_coord, y_coord)




        next_board = self.observe(self.agent_selection)['observation']
        self.board_history = np.dstack(next_board, self.board_history[:, :, :32])       #update board history

        if self.num_steps>2:
            if np.all(next_board[:,:,(current_index+1)%2] == np.zeros(shape=(16*16))):  #game over when opponent has no particles on board
                game_over = True

        if game_over:
            self.terminations = {name: True for name in self.agents}
            win_reward = 1
            lose_reward = -1
            if current_agent == 'p1_team_a' or 'p3_team_a':
                self.rewards['p1_team_a'] = win_reward
                self.rewards['p3_team_a'] = win_reward
                self.rewards['p2_team_b'] = lose_reward
                self.rewards['p4_team_b'] = lose_reward
            else:
                self.rewards['p1_team_a'] = lose_reward
                self.rewards['p3_team_a'] = lose_reward
                self.rewards['p2_team_b'] = win_reward
                self.rewards['p4_team_b'] = win_reward     

        
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        self.num_steps += 1
        if self.render_mode == "human":
            self.render()

    def burst(self, x_coordinate, y_coordinate):                          #checks if neighbouring tile in board and executes reaction burst updates on the board
        current_agent = self.agent_selection
        current_index = self.agents.index(current_agent)
        if x_coordinate>0:
            x_current = x_coordinate-1
            y_current = y_coordinate
            if self.board[x_current, y_current, (current_index)%2] == 0:          
                self.board[x_current, y_current, (current_index)%2] = 1             
                self.board[x_current, y_current, (current_index%2)*3 + 2] = 1    

            elif self.board[x_current, y_current, (current_index%2)*3 + 2] == 1:
                self.board[x_current, y_current, (current_index%2)*3 + 2] = 0
                self.board[x_current, y_current, (current_index%2)*3 + 3] = 1

            elif self.board[x_current, y_current, (current_index%2)*3 + 3] == 1:
                self.board[x_current, y_current, (current_index%2)*3 + 3] = 0
                self.board[x_current, y_current, (current_index%2)*3 + 4] = 1

            elif self.board[x_current, y_current, (current_index%2)*3 + 4] == 1:
                self.board[x_current, y_current, (current_index%2)*3 + 4] = 0
                self.board[x_current, y_current, (current_index)%2] = 0
                self.burst(x_current, y_current)

            elif self.board[x_current, y_current, (current_index+1)%2] == 1:
                self.board[x_current, y_current, (current_index+1)%2] = 0
                self.board[x_current, y_current, (current_index)%2] = 1

                if self.board[x_current, y_current, ((current_index+1)%2)*3 + 2] == 1:
                    self.board[x_current, y_current, ((current_index+1)%2)*3 + 2] = 0
                    self.board[x_current, y_current, (current_index%2)*3 + 3] = 1

                elif self.board[x_current, y_current, ((current_index+1)%2)*3 + 3] == 1:
                    self.board[x_current, y_current, ((current_index+1)%2)*3 + 3] = 0
                    self.board[x_current, y_current, (current_index%2)*3 + 4] = 1

                elif self.board[x_current, y_current, ((current_index+1)%2)*3 + 4] == 1:
                    self.board[x_current, y_current, ((current_index+1)%2)*3 + 4] = 0
                    self.board[x_current, y_current, (current_index)%2] = 0
                    self.burst(x_current, y_current)



        if x_coordinate<16:
            x_current = x_coordinate+1
            y_current = y_coordinate
            if self.board[x_current, y_current, (current_index)%2] == 0:          
                self.board[x_current, y_current, (current_index)%2] = 1             
                self.board[x_current, y_current, (current_index%2)*3 + 2] = 1    

            elif self.board[x_current, y_current, (current_index%2)*3 + 2] == 1:
                self.board[x_current, y_current, (current_index%2)*3 + 2] = 0
                self.board[x_current, y_current, (current_index%2)*3 + 3] = 1

            elif self.board[x_current, y_current, (current_index%2)*3 + 3] == 1:
                self.board[x_current, y_current, (current_index%2)*3 + 3] = 0
                self.board[x_current, y_current, (current_index%2)*3 + 4] = 1

            elif self.board[x_current, y_current, (current_index%2)*3 + 4] == 1:
                self.board[x_current, y_current, (current_index%2)*3 + 4] = 0
                self.board[x_current, y_current, (current_index)%2] = 0
                self.burst(x_current, y_current)

            elif self.board[x_current, y_current, (current_index+1)%2] == 1:
                self.board[x_current, y_current, (current_index+1)%2] = 0
                self.board[x_current, y_current, (current_index)%2] = 1

                if self.board[x_current, y_current, ((current_index+1)%2)*3 + 2] == 1:
                    self.board[x_current, y_current, ((current_index+1)%2)*3 + 2] = 0
                    self.board[x_current, y_current, (current_index%2)*3 + 3] = 1

                elif self.board[x_current, y_current, ((current_index+1)%2)*3 + 3] == 1:
                    self.board[x_current, y_current, ((current_index+1)%2)*3 + 3] = 0
                    self.board[x_current, y_current, (current_index%2)*3 + 4] = 1

                elif self.board[x_current, y_current, ((current_index+1)%2)*3 + 4] == 1:
                    self.board[x_current, y_current, ((current_index+1)%2)*3 + 4] = 0
                    self.board[x_current, y_current, (current_index)%2] = 0
                    self.burst(x_current, y_current)


        if y_coordinate>0:
            x_current = x_coordinate
            y_current = y_coordinate-1
            if self.board[x_current, y_current, (current_index)%2] == 0:          
                self.board[x_current, y_current, (current_index)%2] = 1             
                self.board[x_current, y_current, (current_index%2)*3 + 2] = 1    

            elif self.board[x_current, y_current, (current_index%2)*3 + 2] == 1:
                self.board[x_current, y_current, (current_index%2)*3 + 2] = 0
                self.board[x_current, y_current, (current_index%2)*3 + 3] = 1

            elif self.board[x_current, y_current, (current_index%2)*3 + 3] == 1:
                self.board[x_current, y_current, (current_index%2)*3 + 3] = 0
                self.board[x_current, y_current, (current_index%2)*3 + 4] = 1

            elif self.board[x_current, y_current, (current_index%2)*3 + 4] == 1:
                self.board[x_current, y_current, (current_index%2)*3 + 4] = 0
                self.board[x_current, y_current, (current_index)%2] = 0
                self.burst(x_current, y_current)

            elif self.board[x_current, y_current, (current_index+1)%2] == 1:
                self.board[x_current, y_current, (current_index+1)%2] = 0
                self.board[x_current, y_current, (current_index)%2] = 1

                if self.board[x_current, y_current, ((current_index+1)%2)*3 + 2] == 1:
                    self.board[x_current, y_current, ((current_index+1)%2)*3 + 2] = 0
                    self.board[x_current, y_current, (current_index%2)*3 + 3] = 1

                elif self.board[x_current, y_current, ((current_index+1)%2)*3 + 3] == 1:
                    self.board[x_current, y_current, ((current_index+1)%2)*3 + 3] = 0
                    self.board[x_current, y_current, (current_index%2)*3 + 4] = 1

                elif self.board[x_current, y_current, ((current_index+1)%2)*3 + 4] == 1:
                    self.board[x_current, y_current, ((current_index+1)%2)*3 + 4] = 0
                    self.board[x_current, y_current, (current_index)%2] = 0
                    self.burst(x_current, y_current)


        if y_coordinate<16:
            x_current = x_coordinate
            y_current = y_coordinate+1
            if self.board[x_current, y_current, (current_index)%2] == 0:          
                self.board[x_current, y_current, (current_index)%2] = 1             
                self.board[x_current, y_current, (current_index%2)*3 + 2] = 1    

            elif self.board[x_current, y_current, (current_index%2)*3 + 2] == 1:
                self.board[x_current, y_current, (current_index%2)*3 + 2] = 0
                self.board[x_current, y_current, (current_index%2)*3 + 3] = 1

            elif self.board[x_current, y_current, (current_index%2)*3 + 3] == 1:
                self.board[x_current, y_current, (current_index%2)*3 + 3] = 0
                self.board[x_current, y_current, (current_index%2)*3 + 4] = 1

            elif self.board[x_current, y_current, (current_index%2)*3 + 4] == 1:
                self.board[x_current, y_current, (current_index%2)*3 + 4] = 0
                self.board[x_current, y_current, (current_index)%2] = 0
                self.burst(x_current, y_current)

            elif self.board[x_current, y_current, (current_index+1)%2] == 1:
                self.board[x_current, y_current, (current_index+1)%2] = 0
                self.board[x_current, y_current, (current_index)%2] = 1

                if self.board[x_current, y_current, ((current_index+1)%2)*3 + 2] == 1:
                    self.board[x_current, y_current, ((current_index+1)%2)*3 + 2] = 0
                    self.board[x_current, y_current, (current_index%2)*3 + 3] = 1

                elif self.board[x_current, y_current, ((current_index+1)%2)*3 + 3] == 1:
                    self.board[x_current, y_current, ((current_index+1)%2)*3 + 3] = 0
                    self.board[x_current, y_current, (current_index%2)*3 + 4] = 1

                elif self.board[x_current, y_current, ((current_index+1)%2)*3 + 4] == 1:
                    self.board[x_current, y_current, ((current_index+1)%2)*3 + 4] = 0
                    self.board[x_current, y_current, (current_index)%2] = 0
                    self.burst(x_current, y_current)



    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif self.render_mode == "ansi":
            return str(self.board)
        elif self.render_mode in {"human", "rgb_array"}:
            return self._render_gui()
        else:
            raise ValueError(
                f"{self.render_mode} is not a valid render mode. Available modes are: {self.metadata['render_modes']}"
            )



    def _render_gui(self):
        if self.screen is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.set_caption("CHAIN REACTION")
                self.screen = pygame.display.set_mode(self.BOARD_SIZE)
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface(self.BOARD_SIZE)

        self.screen.blit(self.bg_image, (0, 0))

        for X in range(16*16):

            pos_x = X % 16 * self.cell_size[0]
            pos_y = (
                self.BOARD_SIZE[1] - (X // 16 + 1) * self.cell_size[1]
            )  # offset because pygame display is flipped

            team = 'team_a' if X <16*8 else 'team_b'
            piece_img = self.piece_images[team]
            self.screen.blit(piece_img, (pos_x, pos_y))

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
