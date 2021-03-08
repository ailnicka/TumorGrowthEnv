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

    def __init__(self,
                 mode = None,  # can be different modes for the definition, ie. 'None', '3weeks', 'no_radiation_limit'
                 params_filename: str = None,
                 tumors_list = None,  # when we want to model many tumor types on the same time
                 # tumor_id: int = 1, # when we want just one tumor
                 parallel_runs: int = 1,  # how many parallel runs per tumor - might be different as simulation is stochastic
                 reward_scheme = None,  # None as now, 'binary' for single tumor just binary reward: 1 if cured, 0 if not
                 promotion: int = 100,  # how many more important is reward after 10 days of simulation
                 cycle_in_hours: int = 24):  # timestep of radiotherapy
        if params_filename is None:
            params_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/default-parameters.json")
        if tumors_list is None:
            tumors_list = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/tumor-lib/tumor-{}.txt".format(i))
                           for i in range(1, 11)]
        else:
            tumors_list = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/tumor-lib/tumor-{}.txt".format(i))
                           for i in tumors_list]
        # with tumor_id version
        # tumors_list = [os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                             "data/tumor-lib/tumor-{}.txt".format(tumor_id))]
        params = sim.load_parameters(params_filename)
        tumors = [sim.load_state(tumors, params) for tumors in tumors_list]
        self.experiment = sim.Experiment(params,
                                    tumors,
                                    parallel_runs,  # number of runs per tumor
                                    1) # num protocols
        # trick to get starting situation..
        self.experiment.run(0)
        # values at the beginning
        self.tumor_cells = self.experiment.get_results()[0]  # since we always have 1 protocol at the time, we can drop one list encapsulation
        if not ((reward_scheme is None) or (reward_scheme == 'binary')):
            raise ValueError("Reward scheme can be None or binary!!")
        if not ((mode is None) or (mode == '3weeks') or (mode == 'no_radiation_limit')):
            raise ValueError("Mode can be None or 3weeks or no_radiation_limit!!")
        self.mode = mode
        self.time = 0
        self.cumulative_dose = 0
        if self.mode == '3weeks':
            self.weekly_dose = 0
        self.cycle_in_hours = cycle_in_hours
        # reward definition
        self.start_reward = - np.mean(self.tumor_cells)
        self.reward = 0
        self.reward_scheme = reward_scheme
        self.promotion = promotion
        # gym spaces
        self.observation_space = spaces.MultiDiscrete([1500]*len(tumors_list)*parallel_runs)  # lots of cells allowed per tumor: taken from GA paper
        self.action_space = spaces.MultiDiscrete([self.cycle_in_hours,  # irradiation possible on full hours
                                                 11])  # range between 0-5Gy every 0.5 Gy

    def step(self, action):
        (delay, dose) = action
        if self.mode is None:
            self.cumulative_dose += 0.5 * dose
            # added to limit cumulative dose to 10 Gy
            if self.cumulative_dose > 10:
                dose -= (self.cumulative_dose - 10)
                self.cumulative_dose = 10
            translated_action = (self.time + delay*600, 0.5 * dose)
            self.experiment.add_irradiations([[translated_action]])  # add irradiation
            self.experiment.run(self.cycle_in_hours * 600)  # evolve tumors for cycle of hours
            self.time += self.cycle_in_hours * 600
            self.tumor_cells = self.experiment.get_results()[0]
            # reward is relatively smaller on intermediate steps, but big for the final result
            if self.reward_scheme is None:
                self.reward = self.start_reward - np.mean(self.tumor_cells)
            elif self.reward_scheme == 'binary':
                self.reward = 0
            # finish when time over 5 days or dose over 10 Gy
            done = bool(self.time >= 5 * 24 * 600 or self.cumulative_dose >= 10)
            # if the dose or time exceeded, simulate up to 10 days and promote reward as the final one
            if done:
                leftover_time = 10 * 24 * 600 - self.time
                self.experiment.run(leftover_time)
                self.tumor_cells = self.experiment.get_results()[0]
                if self.reward_scheme is None:
                    self.reward = self.promotion * (self.start_reward - np.mean(self.tumor_cells))
                elif self.reward_scheme == 'binary':
                    self.reward = 1 if np.mean(self.tumor_cells) == 0 else 0

        elif self.mode == '3weeks':
            self.weekly_dose += dose
            # added to limit cumulative weekly dose to 10 Gy
            if self.weekly_dose > 10:
                dose -= (self.weekly_dose - 10)
                self.weekly_dose = 10
            self.cumulative_dose += dose
            translated_action = (self.time + delay * 600, 0.5 * dose)
            self.experiment.add_irradiations([[translated_action]])  # add irradiation
            self.time += 24 * 600  # add time to check if it's not weekend
            if self.time == 5*24*600 or self.time == 12*24*600: # no radiation over weekend
                self.experiment.run(3 * 24 * 600)  # evolve tumors for a weekend
                self.weekly_dose = 0
                self.time += 2 * 24 * 600
            elif self.time == 19*24*600:  # add a week on top after 3rd week
                self.experiment.run(10 * 24 * 600)  # evolve tumors for additional week including 2 weekends
                self.weekly_dose = 0
                self.time += 9 * 24 * 600
            else:
                self.experiment.run(24*600)  # normally run for one day
            self.tumor_cells = self.experiment.get_results()[0]
            # reward is relatively smaller on intermediate steps, but big for the final result
            self.reward = self.start_reward - np.mean(self.tumor_cells)
            # finish after 4 weeks of simulation
            done = bool(self.time >= 28 * 24 * 600)
            # if the time exceeded, promote reward as the final one
            if done:
                self.reward = self.promotion * self.reward

        elif self.mode == 'no_radiation_limit':
            self.cumulative_dose += 0.5 * dose
            translated_action = (self.time + delay * 600, 0.5 * dose)
            self.experiment.add_irradiations([[translated_action]])  # add irradiation
            self.tumor_cells = self.experiment.get_results()[0]
            # reward for less cancer cells, but discounted by the applied dose of radiotherapy
            self.reward = self.start_reward - np.mean(self.tumor_cells) - 10*self.cumulative_dose
            done = bool(self.tumor_cells == 0)

        info = {"delay [h]": delay , "dose [Gy]": dose*0.5, "cumulative_dose": self.cumulative_dose, "leftover_cells": np.mean(self.tumor_cells),
                    "fitness_func": 1500 - np.mean(self.tumor_cells)}
        return np.array(self.tumor_cells).flatten(), self.reward, done, info

    def reset(self):
        self.experiment.reset()
        # reset experiment's parameters
        self.experiment.run(0)
        self.tumor_cells = self.experiment.get_results()[0]
        self.start_reward = - np.mean(self.tumor_cells)
        self.reward = 0
        self.time = 0
        self.cumulative_dose = 0
        if self.mode == '3weeks':
            self.weekly_dose = 0
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
