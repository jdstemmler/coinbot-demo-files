from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bonsai_tools
from simulator_integration import Model

from render import Viewer

from time import sleep

class Star():
    """
    Definition of  State, Terminal, Actions, Reward. 
    For new experiments, you should modify the following methods:

    1. `get_state`: observe new states from the simulator, and define them in an attribute
    2. `get_terminal`: the terminal condition
    3. `set_action`: receive brain action, transform, output action and call simulator.model_step. 
    4. `brain_action_to_sim_action`: transform brain action to the form needed for `set_action`.
    5. `get_reward`: calculate reward from last step
    6. `simulator_reset_config`: get next state and reset configurations.
    """

    def __init__(self, predict=False, render=False):

        self.model = Model(predict=predict, render=render)
        self.simulator_reset_config()
    
    def get_state(self):
        """Get state observations from simulator, transform observations to state sent to BRAIN, executed every iteration.
        """
        self.observations = self.model.simulator_get_observations()
        self.state = dict(
            position = self.observations['position'],
            velocity = self.observations['velocity'],
            angle = self.observations['angle'],
            rotation = self.observations['rotation']
        )

        return self.state

    def get_terminal(self, state):
        """Terminal condition check
        
        Parameters
        ----------
        state : dict
            dictionary containing state variables
        
        Returns
        -------
        boolean
            terminal condition or not (True means terminal)
        """
        
        position_terminal = abs(state['position']) > self.model.simulator.x_threshold
        theta_threshold = abs(state['angle']) > self.model.simulator.theta_threshold_radians

        return position_terminal | theta_threshold | (self.model.iteration > self.model.max_iterations)

    def set_action(self, brain_action):
        """Set action and submit to simulator
        
        Takes brain_action as input, transforms it if necessary for input and sets sim_action to simulator, runs the simulator one step

        Parameters
        ----------
        brain_action : action
        """
        self.brain_action = brain_action
        self.sim_action = self.brain_action_to_sim_action(brain_action)
        self.model.simulator_step(self.sim_action)

    def brain_action_to_sim_action(self, brain_action):
        """Convert brain_action -> sim_action
        Takes brain action and transforms it to sim action (action actually applied to the environment)
        """
        command = brain_action['command']

        brain_to_sim_map = {
            -1: 0,
            1: 1
        }

        return {'command': brain_to_sim_map[command]}

    def get_reward(self, state, terminal):
        """Return reward value
        
        Parameters
        ----------
        state : dict
            State values
        terminal : boolean
            Terminal state
        
        Returns
        -------
        double
            Reward value
        """

        if not terminal:
            reward = 1
        else:
            reward = 0

        return reward

    def simulator_reset_config(self, config=None):
        """Reset simulator configuration
        """

        self.model.simulator_reset(config)
        
        self.state = self.get_state()

        self.initial_action = {'command': np.random.choice([-1, 1])}

        self.brain_action = self.initial_action
        self.sim_action = self.brain_action_to_sim_action(self.brain_action)


    def define_logged_observations(self):
        """Log actions and states.
        Defines the logged_observations dictionary which will be logged in file
        """
        logged_observations = {}
        logged_observations.update(self.observations)
        logged_observations.update(self.state)
        # logged_observations.update(self.config)
        logged_observations.update(bonsai_tools.rename_action(self.brain_action,'brain'))
        logged_observations.update(bonsai_tools.rename_action(self.sim_action,'sim'))		
        return logged_observations

def simple_brain_controller(state):
    # Application Specific Controller specific to House Energy
    action = 1
    if state['angle'] < 0:
        action = -1
    elif state['angle'] > 0:
        action = 1

    return action

if __name__ == "__main__":
    """use star.py as main to test star piece without bonsai platform in the loop
    """
    # TODO: provide some instructions for testing
    print("Testing star.py")
    star = Star()
    star.simulator_reset_config()
    state = star.get_state()
