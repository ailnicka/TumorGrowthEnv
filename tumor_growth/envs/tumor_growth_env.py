import os
import emt6ro.simulation as sim
import numpy as np
import gym
from gym import spaces


class TumorGrowthEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a env where the growth of the tumor is simulated and it is possible to irradiate it.
    Multiple tumors can be simulated simultaneously, since the simulation is stochastic.
    Possible action is to add irradiation daily: the delay on the day and the dose are needed.
    Reward is minus the average of leftover cancer cells.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, params_filename: str = None,
                 tumors_list=None,
                 parallel_runs: int = 10):
        if params_filename is None:
            params_filename = "tumor_growth/envs/data/default-parameters.json"
        if tumors_list is None:
            tumors_list = ["tumor_growth/envs/data/tumor-lib/tumor-{}.txt".format(i) for i in range(1, 11)]
        params = sim.load_parameters(params_filename)
        tumors = [sim.load_state(tumors, params) for tumors in tumors_list]
        self.experiment = sim.Experiment(params,
                                    tumors,
                                    parallel_runs,  # num_tests
                                    1) # num protocols
        self.experiment.run(0)
        self.tumor_cells = self.experiment.get_results()
        self.reward = - np.mean(self.cumulative_dose)
        self.time = 0
        self.cumulative_dose = 0
        self.observation_space = spaces.MultiDiscrete([1000000]*parallel_runs)  # lots of cells allowed per tumor?
        self.action_space = spaces.Dict({"delay": spaces.Discrete(12),
                                         "dose": spaces.Box(low=0, high=5, shape=(), dtype=np.float32)})

    def step(self, action=None):
        if action is not None:
            translated_action = [(self.time + action.get("delay")*600, action.get("dose"))]
            self.cumulative_dose = action.get("dose")
            self.experiment.add_irradiations([translated_action])  # add irradiation
        self.experiment.run(12 * 600)  # evolve tumors for 12 hours
        self.time += 12 * 600
        self.tumor_cells = self.experiment.get_results()
        self.reward = - np.mean(self.tumor_cells)
        done = (self.reward == 0)
        info = {"cumulative_dose": self.cumulative_dose}
        return self.tumor_cells, self.reward, done, info

    def reset(self):
        self.experiment.reset()

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print("For {} tumors, there are {} cancer cells left in each.".format(
            len(self.tumor_cells), self.tumor_cells))
        print("Average number of leftover cancer cells per tumor is: {}".format(-self.reward))
        print("Radiation dose applied so far: {}".format(self.cumulative_dose))

    def close(self):
        pass
