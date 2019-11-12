import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, csv, sys, time, argparse

from render import Viewer
from time import sleep

# Import the simulator object
from sim.cartpole import CartPole as SIMULATION

def compute_config_params(config):
    calc_config = {
        'total_mass': round(config['masspole'] + config['masscart'], 4),
        'polemass_length': round((config['masspole'] * config['length']), 4)
    }

    return calc_config

class Model:
    """
    Instantiation of simulator for training with Bonsai. 
    For new simulators, you should modify the following methods:
    1. `simulator_initialize`: define any parameters here for the start of your simulator, i.e., 
        parameters that will persist throughout all episodes of training. 
        The number of arguments here should hopefully minimal. 
        The last call in this method should be to `simulator_reset` -> 
    2. `simulator_reset`: this method will be called at the start of every new episode. 
        Use this method to reset variables/parameters created in previous episode that are no longer needed. 
        The only arguments here should be a dictionary named `config` 
        which will be passed to `simulator_config` below, and any simulator specific variable. 
        This method should also call `simulator_config` ->
    3. `simulator_config`: this method is called by `simulator_reset`. 
        This is where you define state parameters specific for the new episode. 
        Pass the `config` argument here to define state parameters for the simulator. 
        This method should end by calling the simulator specified in the `sim` directory:
        TODO: define how to utilize _inkling_ variables in  `simulator_config` 
    4. `simulator_step`: this method is called at every step. 
        It should take an argument named `action`, which will take the action from `star.set_action`. 
        This method should also update any necessary state variables from state.
    5. `simulator_get_observations`: Return observations from simulator. 
        Define all state variables you want to return. Note, these are not the 
        variables that are necessarily consumed by the state of your RL system. 
        Those are defined star.py. Instead, these are variables you want to log.
    6. `simulator_step`: Complete one step. 
        Should call `simulator_set_action` once.

    """

    def __init__(self, predict=False, render = False):
        """Initialize the cartpole with some default values.
        """
        self.render = render
        if self.render:
            self.viewer = Viewer()

        # set the model directory
        self.modeldir = 'sim'
        os.chdir('./' +  self.modeldir)
        print("Using simulator file from: ", os.getcwd())

        # initial state of the cart
        self.cart_config = {
            'masscart': 1.,
            'masspole': 0.1,
            'length'  : .5,
            'force_mag': 10
        }
        self.cart_config.update(compute_config_params(self.cart_config))

        if predict:
            print("Model is in PREDICT mode")
            self.predict = predict
            self.max_iterations = 5000
        else:
            print("Model is in TRAINING mode")
            self.predict = predict
            self.max_iterations = 500

        self.simulator_initialize()

    def simulator_initialize(self):
        """
        Initialize simulator environment and with initial conditions once, at launch.
        
        Parameters
        ----------
        """

        self.simulator_reset()

    def simulator_reset(self, config=None):
        """Resets the simulator, executed at every episode start.
        
        Parameters
        ----------
        config : dictionary
            Configuration parameters to be passed on to `simulator_configure`
        
        """
        self.iteration = 0

        if config is None:
            config = getattr(self, 'cart_config', None)
        
        if config is None:
            config = {
                'masscart': self.simulator.masscart,
                'masspole': self.simulator.masspole,
                'length'  : self.simulator.length,
                'force_mag': self.simulator.force_mag
            }
        
        if self.predict:
            config['masscart'] = round(np.random.random() + .5, 3)
            config['masspole'] = round(np.random.random() * .1 + .05, 4)
            pass

        config.update(compute_config_params(config))

        self.cart_config = config

        if self.predict:
            print(self.cart_config)

        self.simulator_configure(config)

    def simulator_configure(self, config):
        """Set up simulator initial conditions and configuration at the start of every episode
        
        Parameters
        ----------
        TODO: assign config params to sim config variables and assign them from `star.simulator_reset_config`t st
        predict : bool, optional
            Whether to run train or eval loop (the default is False, which invokes training loop)
        
        """

        self.simulator = SIMULATION()

        if self.render:
            self.viewer.model = self.simulator
            self.viewer.update()

        if config is not None:
            for key, value in config.items():
                setattr(self.simulator, key, value)
        # print(f'Mass Cart: {self.simulator.masscart}, Mass Pole: {self.simulator.masspole}')
        
    def simulator_get_observations(self) -> dict:
        """Return observations from simulator, executed every iteration.
        
        Returns
        -------
        dictionary
            Dictionary of state elements at current iteration
        """
        sim_state = self.simulator.state
        observations = {
            'position': sim_state.x,
            'velocity': sim_state.x_dot,
            'angle': sim_state.y,
            'rotation': sim_state.y_dot
        }

        return observations

    def simulator_set_action(self, action: dict):
        """Execute action provided.
        
        Parameters
        ----------
        action : dictionary
            Which action(s) to execute
        
        """
        self.simulator.step(action['command'])

    def simulator_step(self, action):
        """Execute simulator for one time step.
        
        Parameters
        ----------
        action : action
            Action to execute through `simulator_set_action`
        """
        self.simulator_set_action(action)
        if self.render:
            self.viewer.update()
            # sleep(.05)

        self.iteration += 1

def simple_controller(state):
    action = 0
    if state['angle'] > 0:
        action = 1
    
    return {'command': action}

if __name__ == "__main__":
    """use simulator_integration.py as main to test simulator_integration piece alone
    """
    print("Running simulator_integration.py")
    model = Model(render=True, predict=True)
    for i in range(10):
        model.simulator_reset()
        print(model.cart_config)
        is_not_terminal = True
        
        while is_not_terminal:
            state = model.simulator_get_observations()
            action = simple_controller(state)
            model.simulator_step(action)
            is_not_terminal = (
                (abs(state['position']) <= model.simulator.x_threshold) & 
                (abs(state['angle']) <= model.simulator.theta_threshold_radians)
            )