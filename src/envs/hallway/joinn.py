from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.multiagentenv import MultiAgentEnv

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging
import random


class JoinNEnv(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """
    def __init__(
            self,
            key='none',
            n_agents=5,
            n_groups=2,
            state_numbers=[4,4,4,4,4],
            group_ids=[0,0,1,1,1],
            reward_win=10,
            obs_last_action=False,
            state_last_action=True,
            seed=None
    ):
        # Map arguments
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        self.n_agents = n_agents
        self.n_states = np.array(state_numbers,
                                 dtype=np.int)

        # Observations and state
        self.obs_last_action = obs_last_action
        self.state_last_action = state_last_action

        # Rewards args
        self.reward_win = reward_win

        # Other
        self._seed = seed

        # Actions
        self.n_actions = 3
        self.n_groups = n_groups
        self.group_ids = np.array(group_ids)

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        self.episode_limit = max(state_numbers) + 10

        # initialize agents
        self.state_n = np.array([np.random.randint(low=1, high=self.n_states[i]+1) for i in range(self.n_agents)],
                                dtype=np.int)

        # self.group_members = [[] for _ in range(self.n_groups)]
        # self.status_by_group = [[] for _ in range(self.n_groups)]
        # for agent_i, group_i in enumerate(self.group_ids):
        #     self.group_members[group_i].append(agent_i)
        #     self.status_by_group[group_i].append(0)

        self.active_group = [True for _ in range(self.n_groups)]
        self.active_agent = np.array([True for _ in range(self.n_agents)])
        self._win_group = 0
        self._fail_group = 0

    def step(self, actions):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        last_state = self.state_n

        for agent_i, action in enumerate(actions):
            if self.active_agent[agent_i]:
                if action == 0:
                    pass
                elif action == 1:
                    self.state_n[agent_i] = max(0, self.state_n[agent_i] - 1)
                elif action == 2:
                    self.state_n[agent_i] = min(self.n_states[agent_i], self.state_n[agent_i] + 1)

        reward = 0
        terminated = False
        info['battle_won'] = False

        win_in_this_round = 0
        win_agents = np.array([False for _ in range(self.n_agents)])

        for group_i in range(self.n_groups):
            if self.active_group[group_i]:
                id = self.state_n[self.group_ids == group_i]

                if (id == 0).all():
                    reward += self.reward_win
                    self._win_group += 1
                    self.active_group[group_i] = False
                    self.active_agent[self.group_ids == group_i] = False
                    win_agents[self.group_ids == group_i] = True
                    win_in_this_round += 1
                elif (id == 0).any():
                    self.active_group[group_i] = False
                    self.active_agent[self.group_ids == group_i] = False
                    self._fail_group += 1

        info['win_group'] = self._win_group

        if win_in_this_round > 1:
            self._win_group -= win_in_this_round
            reward -= self.reward_win * 1.5 * win_in_this_round
            self.active_agent[win_agents] = True
            self.state_n[win_agents] = last_state[win_agents]

        if self._win_group == self.n_groups:
            terminated = True
            self.battles_won += 1
            info['battle_won'] = True
        elif self._fail_group == self.n_groups:
            terminated = True

        if self._episode_steps >= self.episode_limit:
            terminated = True

        if terminated:
            self._episode_count += 1
            self.battles_game += 1

        return reward, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return np.array([self.state_n[agent_id], float(self.active_agent[agent_id])])

    def get_obs_size(self):
        """Returns the size of the observation."""
        return 2

    def get_state(self):
        """Returns the global state."""
        return np.concatenate(self.get_obs())

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.n_agents * self.get_obs_size()

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return [1] * self.n_actions

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        self._episode_steps = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.state_n = np.array([np.random.randint(low=1, high=self.n_states[i]+1) for i in range(self.n_agents)],
                                dtype=np.int)
        self.active_group = [True for _ in range(self.n_groups)]
        self.active_agent = np.array([True for _ in range(self.n_agents)])
        self._win_group = 0
        self._fail_group = 0

        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "win_rate": self.battles_won / self.battles_game
        }
        return stats

    def clean(self):
        self.p_step = 0
        self.rew_gather = []
        self.is_print_once = False