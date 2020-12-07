import emt6ro.simulation as sim
import numpy as np
import gym
from gym import spaces
import os


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
                 parallel_runs: int = 1):
        if params_filename is None:
            params_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/default-parameters.json")
        if tumors_list is None:
            tumors_list = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/tumor-lib/tumor-{}.txt".format(i))
                           for i in range(1, 11)]
        params = sim.load_parameters(params_filename)
        tumors = [sim.load_state(tumors, params) for tumors in tumors_list]
        self.experiment = sim.Experiment(params,
                                    tumors,
                                    parallel_runs,  # num_tests
                                    1) # num protocols
        # trick to get starting situation..
        self.experiment.run(0)
        # values at the beginning
        self.tumor_cells = self.experiment.get_results()
        self.reward = - np.mean(self.tumor_cells)
        self.time = 0
        self.cumulative_dose = 0
        # gym spaces
        self.observation_space = spaces.MultiDiscrete([1000000]*len(tumors_list)*parallel_runs)  # lots of cells allowed per tumor?
        self.action_space = spaces.Dict({"delay": spaces.Discrete(12),  # irradiation possible on full hours
                                         "dose": spaces.Discrete(11)})  # range between 0-5Gy every 0.5 Gy

    def step(self, action):
        print(action)
        translated_action = [self.time + action.get("delay")*600, 0.5 * action.get("dose")]
        print("tr action", translated_action)
        self.cumulative_dose += 0.5 * action.get("dose")
        self.experiment.add_irradiations([translated_action])  # add irradiation
        self.experiment.run(12 * 600)  # evolve tumors for 12 hours
        self.time += 12 * 600
        self.tumor_cells = self.experiment.get_results()
        self.reward = - np.mean(self.tumor_cells)
        done = bool(self.reward == 0)
        info = {"cumulative_dose": self.cumulative_dose}
        return (np.array(self.tumor_cells)).flatten(), self.reward, done, info

    def reset(self):
        self.experiment.reset()
        # reset experiment's parameters
        self.experiment.run(0)
        self.tumor_cells = self.experiment.get_results()
        self.reward = - np.mean(self.tumor_cells)
        self.time = 0
        self.cumulative_dose = 0
        print(self.tumor_cells)
        return (np.array(self.tumor_cells)).flatten()

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print("For {} tumors, there are {} cancer cells left in each.".format(
            len(self.tumor_cells), self.tumor_cells))
        print("Average number of leftover cancer cells per tumor is: {}".format(-self.reward))
        print("Radiation dose applied so far: {}".format(self.cumulative_dose))

    def close(self):
        pass
