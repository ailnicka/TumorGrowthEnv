import emt6ro.simulation as sim
import numpy as np
import gym
from gym import spaces
import os

# the timestep of the radiotherapy
CYCLE_IN_HOURS = 24
# promotion for the final amount of tumor cells
PROMOTION = 100

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
                 tumors_list = None,  # when we want to model many tumor types on the same time
                 # tumor_id: int = 1, # when we want just one tumor
                 parallel_runs: int = 1):
        if params_filename is None:
            params_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/default-parameters.json")
        if tumors_list is None:
            tumors_list = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/tumor-lib/tumor-{}.txt".format(i))
                           for i in range(1, 11)]
        # with tumor_id version
        # tumors_list = [os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                             "data/tumor-lib/tumor-{}.txt".format(tumor_id))]
        params = sim.load_parameters(params_filename)
        tumors = [sim.load_state(tumors, params) for tumors in tumors_list]
        self.experiment = sim.Experiment(params,
                                    tumors,
                                    parallel_runs,  # num_tests
                                    1) # num protocols
        # trick to get starting situation..
        self.experiment.run(0)
        # values at the beginning
        self.tumor_cells = self.experiment.get_results()[0]  # since we always have 1 protocol at the time, we can drop one list encapsulation
        self.start_reward = - np.mean(self.tumor_cells)
        self.reward = 0
        self.time = 0
        self.cumulative_dose = 0
        # gym spaces
        self.observation_space = spaces.MultiDiscrete([1500]*len(tumors_list)*parallel_runs)  # lots of cells allowed per tumor: taken from GA paper
        self.action_space = spaces.MultiDiscrete([CYCLE_IN_HOURS,  # irradiation possible on full hours
                                                 11])  # range between 0-5Gy every 0.5 Gy

    def step(self, action):
        (delay, dose) = action
        self.cumulative_dose += 0.5 * dose
        # added to limit cumulative dose to 10 Gy
        if self.cumulative_dose > 10:
            dose -= (self.cumulative_dose - 10)
            self.cumulative_dose = 10
        translated_action = (self.time + delay*600, 0.5 * dose)
        self.experiment.add_irradiations([[translated_action]])  # add irradiation
        self.experiment.run(CYCLE_IN_HOURS * 600)  # evolve tumors for cycle of hours
        self.time += CYCLE_IN_HOURS * 600
        self.tumor_cells = self.experiment.get_results()[0]
        # reward is relatively smaller on intermediate steps, but big for the final result
        self.reward = self.start_reward - np.mean(self.tumor_cells)
        # finish when no leftover cancer cells or time over five days or dose over 10 Gy
        done = bool(self.time >= 5 * 24 * 600 or self.cumulative_dose >= 10)
        # if the dose or time exceeded, simulate up to 10 days and promote reward as the final one
        if done:
            leftover_time = 10 * 24 * 600 - self.time
            self.experiment.run(leftover_time)
            self.tumor_cells = self.experiment.get_results()[0]
            self.reward = PROMOTION * (self.start_reward - np.mean(self.tumor_cells))
        info = {"cumulative_dose": self.cumulative_dose, "leftover_cells": np.mean(self.tumor_cells), "fitness_func": 1500 - np.mean(self.tumor_cells)}
        return np.array(self.tumor_cells).flatten(), self.reward, done, info

    def reset(self):
        self.experiment.reset()
        # reset experiment's parameters
        self.experiment.run(0)
        self.tumor_cells = self.experiment.get_results()[0]
        self.reward = - np.mean(self.tumor_cells)
        self.time = 0
        self.cumulative_dose = 0
        return np.array(self.tumor_cells).flatten()

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print("For {} tumors in {} simulation runs, there are {} cancer cells left in each.".format(
            len(self.tumor_cells), len(self.tumor_cells[0]), self.tumor_cells))
        print("Average number of leftover cancer cells per tumor is: {}".format(np.mean(self.tumor_cells)))
        print("Radiation dose applied so far: {}".format(self.cumulative_dose))

    def close(self):
        pass
