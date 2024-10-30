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
        "render_modes": ["human", "ansi", "rgb_array"],
        "name": "ChainReaction_v0",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(self, render_mode: str = "human", screen_height: int = 800):
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

        if current_index in [0,2]:
            legal_moves = np.where(np.flatten(observation[:,:,1])==0)
        if current_index in [1,3]:
            legal_moves =  np.where(np.flatten(observation[:,:,0])==0)


        action_mask = np.zeros(16*16, "int8")
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}
    

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

        self.board = np.zeros(shape=(16,16,8))

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.board_history = np.zeros((16, 16, 32), dtype=bool)

        if self.render_mode == "human":
            self.render()


    def step(self, action):# unfinished
        '''
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        self._go = self._go.play_move(coords.from_flat(action))
        self._last_obs = self.observe(self.agent_selection)
        current_agent_plane, opponent_agent_plane = self._encode_board_planes(
            self.agent_selection
        )
        self.board_history = np.dstack(
            (current_agent_plane, opponent_agent_plane, self.board_history[:, :, :-2])
        )

        if self._go.is_game_over():
            self.terminations = self._convert_to_dict(
                [True for _ in range(self.num_agents)]
            )
            self.rewards = self._convert_to_dict(
                self._encode_rewards(self._go.result())
            )
            self.next_legal_moves = [self._N * self._N]
        else:
            self.next_legal_moves = self._encode_legal_actions(
                self._go.all_legal_moves()
            )
        '''
        next_player = self._agent_selector.next()
        self.agent_selection = (
            next_player if next_player else self._agent_selector.next()
        )
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()


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