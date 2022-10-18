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
    """
    metadata = {'render.modes': ['console']}

    def __init__(self,
                reward_type = None, # None is based on the avarage number of leftowver cells, 'kill_prob' is based on how many simulations have killed all tumor cells
                mode = None,  # can be different modes for the definition, ie. 'None', '3weeks', 'no_radiation_limit'
                params_filename: str = None,
                tumors_list = None,  # when we want to model many tumor types on the same time
                # tumor_id: int = 1, # when we want just one tumor
                parallel_runs: int = 1,  # how many parallel runs per tumor - might be different as simulation is stochastic
                promotion: int = 100,  # how many more important is reward after 10 days of simulation
                cycle_in_hours: int = 24,  # timestep of radiotherapy
                with_time: bool = False):  # add day into state
        super(TumorGrowthEnv, self).__init__()

        MODES = [None, '3weeks', 'no_radiation_limit', '2doses']
        REW = [None, 'kill_prob']
        if not mode in MODES:
            raise ValueError(f"Mode must be one of {MODES}!!")
        if not reward_type in REW:
            raise ValueError(f"Reward must be one of {REW}!!")
        self.mode = mode
        self.with_time = with_time

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
        self.rewad_type = reward_type
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
        self.time = 0
        self.cumulative_dose = 0
        if self.mode == '3weeks':
            self.weekly_dose = 0
        self.cycle_in_hours = cycle_in_hours
        # reward definition
        self.start_cells = np.mean(self.tumor_cells)
        self.reward = 0
        self.promotion = promotion
        # gym spaces
        if with_time:
            maxday = 10 if self.mode != '3weeks' else 28
            self.observation_space = spaces.MultiDiscrete([1500]+[maxday])  # lots of cells allowed per tumor: taken from GA paper
        else:
            self.observation_space = spaces.MultiDiscrete([1500])  # lots of cells allowed per tumor: taken from GA paper
        if self.mode == '2doses':
            self.action_space = spaces.MultiDiscrete([self.cycle_in_hours, self.cycle_in_hours, # irradiation possible on full hours twice a day
                                                      11,  # range between 0-5Gy every 0.5 Gy for daily dose
                                                      3])  # division of daily dose (0:1, 1:3, 1:1)
        else:
            self.action_space = spaces.MultiDiscrete([self.cycle_in_hours,  # irradiation possible on full hours
                                                 11])  # range between 0-5Gy every 0.5 Gy

    def step(self, action):
        if self.mode is None:
            done, info = self._step_default(action)
        elif self.mode == '2doses':
            done, info = self._step_2doses(action)
        elif self.mode == '3weeks':
            done, info = self._step_3weeks(action)
        elif self.mode == 'no_radiation_limit':
            done, info = self._step_no_rad_limit(action)
        if self.with_time:
            day = self.time / 24 / 600  # translate time into dayx
            state = np.array([int(np.mean(self.tumor_cells))]+[day])
        else:
            state = np.array([int(np.mean(self.tumor_cells))])
        return state, self.reward, done, info

    def reset(self):
        self.experiment.reset()
        # reset experiment's parameters
        self.experiment.run(0)
        self.tumor_cells = self.experiment.get_results()[0]
        self.start_cells = np.mean(self.tumor_cells)
        self.reward = 0
        self.time = 0
        self.cumulative_dose = 0
        if self.mode == '3weeks':
            self.weekly_dose = 0
        if self.with_time:
            state = np.array([int(np.mean(self.tumor_cells))]+[0])
        else:
            state = np.array([int(np.mean(self.tumor_cells))])
        return state

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print("For {} tumors in {} simulation runs, there are {} cancer cells left in each.".format(
            len(self.tumor_cells), len(self.tumor_cells[0]), self.tumor_cells))
        print("Average number of leftover cancer cells per tumor is: {}".format(np.mean(self.tumor_cells)))
        print("Simmulations with all cancer cells killed : {}".format(np.sum(self.tumor_cells.flatten() == 0)))
        print("Probability if killed cancer: {}".format(np.sum(self.tumor_cells.flatten() == 0)/len(self.tumor_cells.flatten())))
        print("Radiation dose applied so far: {}".format(self.cumulative_dose))
        print("The simulation was evolved for {} days.".format(int(self.time/24/600)))

    def close(self):
        pass

    def _step_2doses(self, action):
        (delay1, delay2, dose, division) = action

        dose = 0.5 * dose
        dose = min(dose, 10 - self.cumulative_dose)

        self.cumulative_dose += dose

        dose1 = dose * division / 4
        dose2 = dose - dose1

        translated_action = [(self.time + delay1 * 600, dose1), (self.time + delay2 * 600, dose2)]
        self.experiment.add_irradiations([translated_action])  # add irradiations
        self.experiment.run(self.cycle_in_hours * 600)  # evolve tumors for cycle of hours
        self.time += self.cycle_in_hours * 600
        self.tumor_cells = self.experiment.get_results()[0]
        # finish when time over 5 days or dose over 10 Gy
        done = bool(self.time >= 5 * 24 * 600 or self.cumulative_dose >= 10 or self.tumor_cells.all() == 0)
        # if the dose or time exceeded, simulate up to 10 days and promote reward as the final one
        if done:
            leftover_time = 10 * 24 * 600 - self.time
            self.experiment.run(leftover_time)
            self.tumor_cells = self.experiment.get_results()[0]
            self.time = 10 * 24 * 600
            self._update_reward(promotion=True)
        else:
            self._update_reward()

        info = {"delay1": delay1, "delay2": delay2, "dose1": dose1, "dose2": dose2,
                "cumulative_dose": self.cumulative_dose, "leftover_cells": np.mean(self.tumor_cells),
                "killed_cancer_prob": np.sum(self.tumor_cells.flatten() == 0)/len(self.tumor_cells.flatten()),
                "fitness_func": 1500 - np.mean(self.tumor_cells)}
        return done, info

    def _step_default(self, action):
        (delay, dose) = action
        dose = 0.5 * dose
        # added to limit cumulative dose to 10 Gy
        dose = min(dose, 10 - self.cumulative_dose)
        self.cumulative_dose += dose
        translated_action = (self.time + delay * 600, dose)
        self.experiment.add_irradiations([[translated_action]])  # add irradiation
        self.experiment.run(self.cycle_in_hours * 600)  # evolve tumors for cycle of hours
        self.time += self.cycle_in_hours * 600
        self.tumor_cells = self.experiment.get_results()[0]
        # finish when time over 5 days or dose over 10 Gy
        done = bool(self.time >= 5 * 24 * 600 or self.cumulative_dose >= 10 or self.tumor_cells.all() == 0)
        # if the dose or time exceeded, simulate up to 10 days and promote reward as the final one
        if done:
            leftover_time = 10 * 24 * 600 - self.time
            self.experiment.run(leftover_time)
            self.tumor_cells = self.experiment.get_results()[0]
            self.time = 10 * 24 * 600
            self._update_reward(promotion=True)
        else:
            self._update_reward()
        info = {"delay [h]": delay , "dose [Gy]": dose, "cumulative_dose": self.cumulative_dose,
                "leftover_cells": np.mean(self.tumor_cells), "killed_cancer_prob": np.sum(self.tumor_cells.flatten() == 0)/len(self.tumor_cells.flatten()),
                "fitness_func": 1500 - np.mean(self.tumor_cells)}
        return done, info

    def _step_3weeks(self, action):
        (delay, dose) = action
        dose = 0.5 * dose
        # added to limit cumulative weekly dose to 10 Gy
        dose = min(dose, 10 - self.weekly_dose)
        self.weekly_dose += dose
        self.cumulative_dose += dose
        translated_action = (self.time + delay * 600, dose)
        self.experiment.add_irradiations([[translated_action]])  # add irradiation
        self.time += 24 * 600  # add time to check if it's not weekend
        if self.time == 5 * 24 * 600 or self.time == 12 * 24 * 600:  # no radiation over weekend
            self.experiment.run(3 * 24 * 600)  # evolve tumors for a weekend
            self.weekly_dose = 0
            self.time += 2 * 24 * 600
        elif self.time == 19 * 24 * 600:  # add a week on top after 3rd week
            self.experiment.run(10 * 24 * 600)  # evolve tumors for additional week including 2 weekends
            self.weekly_dose = 0
            self.time += 9 * 24 * 600
        else:
            self.experiment.run(24 * 600)  # normally run for one day
        self.tumor_cells = self.experiment.get_results()[0]
        # finish after 4 weeks of simulation
        done = bool(self.time >= 28 * 24 * 600 or self.tumor_cells.all() == 0)
        # if the time exceeded, promote reward as the final one
        if done:
            self._update_reward(promotion=True)
        else:
            self._update_reward()
        info = {"delay [h]": delay, "dose [Gy]": dose, "cumulative_dose": self.cumulative_dose,
                "leftover_cells": np.mean(self.tumor_cells), "killed_cancer_prob": np.sum(self.tumor_cells.flatten() == 0)/len(self.tumor_cells.flatten()),
                "fitness_func": 1500 - np.mean(self.tumor_cells)}
        return done, info

    def _step_no_rad_limit(self, action):
        (delay, dose) = action
        dose = 0.5 * dose
        self.cumulative_dose += dose
        translated_action = (self.time + delay * 600, dose)
        self.experiment.add_irradiations([[translated_action]])  # add irradiation
        self.experiment.run(24 * 600)
        self.time += 24 * 600
        self.tumor_cells = self.experiment.get_results()[0]
        # reward for less cancer cells, but discounted by the applied dose of radiotherapy
        done = bool(self.tumor_cells.all() == 0)
        if done:
            self._update_reward(promotion=True)
        else:
            self._update_reward()

        info = {"delay [h]": delay, "dose [Gy]": dose, "cumulative_dose": self.cumulative_dose,
            "leftover_cells": np.mean(self.tumor_cells), "killed_cancer_prob": np.sum(self.tumor_cells.flatten() == 0)/len(self.tumor_cells.flatten()),
            "fitness_func": 1500 - np.mean(self.tumor_cells)}
        return done, info

    def _update_reward(self, promotion=False):
        if self.rewad_type is None and self.mode != 'no_radiation_limit':
            self.reward = self.start_cells - np.mean(self.tumor_cells) if promotion==False \
                else self.promotion * (self.start_cells - np.mean(self.tumor_cells))
        elif self.reward_type == None and self.mode == 'no_radiation_limit':
            self.reward = (self.type - np.mean(self.tumor_cells) - self.cumulative_dose) if promotion == False \
                else self.promotion * (self.start_cells - np.mean(self.tumor_cells) - self.cumulative_dose)
        elif self.rewad_type == 'kill_prob':
            self.reward = np.sum(self.tumor_cells.flatten() == 0) if promotion == False \
                else self.promotion * np.sum(self.tumor_cells.flatten() == 0)

