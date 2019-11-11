""" This template implements the star.py <> ModelTrainer event pattern with the Matlab Python Engine.
python hub.py --train-brain=brain_name
start python hub.py --brain sh1
"""

import sys, argparse, logging, datetime, time, os, io
import numpy as np
import random
import psutil
from star import Star
from random import randint
from bonsai_ai import Brain, Config, Simulator
from bonsai_ai.logger import Logger
from bonsai_ai import EpisodeStartEvent, SimulateEvent, \
	EpisodeFinishEvent, FinishedEvent, UnknownEvent

import bonsai_tools

log = Logger()
import argparse

def _parse_args():
	parser = argparse.ArgumentParser(
		description='bits for dev')
	parser.add_argument('--log-iterations', action='store_true')
	parser.add_argument('--log-training-speed', action='store_true')
	parser.add_argument('--render', action='store_true')
	args, unknown = parser.parse_known_args()
	return args

log_iterations = _parse_args().log_iterations
monitor_training_speed = _parse_args().log_training_speed
render = _parse_args().render
print('log iterations: {}'.format(log_iterations))
print('log training speed: {}'.format(monitor_training_speed))
print('render simulation: {}'.format(render))

class ModelConnector(Simulator):

	def __init__(self, brain, name, config):
		super(ModelConnector, self).__init__(brain, name)
		self.results_logger = None
		if log_iterations is True:
			self.results_logger = bonsai_tools.log_initialize(brain, pathname='./log/')
		if monitor_training_speed is True:
			monitoring_logger = bonsai_tools.log_initialize(brain, pathname='./log/', log_training_speed = True)
			logged_observations_dict = {'datetime':None,'num_of_sims':None,'iterations':None,'num_iterations_per_s':None}
			bonsai_tools.log_observations_columns(monitoring_logger, logged_observations_dict)
			bonsai_tools.monitor_training(monitoring_logger,logged_observations_dict, brain.name)
	
		self.config = config
		pass

class ModelTrainer(object):

	def __init__(self, sim, predict=False):
		self._sim = sim
		self.episode_count = 0
		self.reset_iteration_metrics()
		self.star = Star(predict=predict, render=render)
		self.star.logger = sim.results_logger
		if log_iterations is True:
			self.logged_observations = self.star.define_logged_observations()
			self.logged_observations = self.update_logged_observations(self.logged_observations)
			bonsai_tools.log_observations_columns(self.star.logger, self.logged_observations)

	def episode_start(self,event):
		if getattr(self._sim, 'sim_id', -1) == -1:
			self.sim_id = self._sim._impl._sim_id
			#if self.sim_id != -1:
			#   print('SimID', self.sim_id)
			
		self.start_episode()
		self.episode_count += 1
		# Check https://docs.bons.ai/references/library-reference.html#event-class for SDK event class documentation from Product
		event.initial_state = self.star.get_state()
		event.terminal = self.star.get_terminal(event.initial_state)
		event.reward = 0 #the initial reward is an arbitrary value since there are no actions taken by BRAIN in initial state
		if log_iterations is True:
			self.logged_observations = self.star.define_logged_observations()
			self.logged_observations = self.update_logged_observations(self.logged_observations)
			bonsai_tools.log_iteration(self.star.logger, self.logged_observations)

	def run(self):
		event = self._sim.get_next_event()

		if isinstance(event, EpisodeStartEvent):
			log.event("Episode Start Train")
			self.episode_start(event)

		# Receive the action from the BRAIN as event.action, run the simulation one step and return the state, action, and reward to the BRAIN. 
		elif isinstance(event, SimulateEvent):
			log.event("Simulate")
			self.iteration_count += 1
			self.action = event.action
			self.star.set_action(self.action)
			event.state = self.star.get_state() 
			event.terminal = self.star.get_terminal(event.state)
			event.reward = self.star.get_reward(event.state, event.terminal)
			#print(event.state)
			self.reward = event.reward
			self.terminal = event.terminal
			self.episode_reward += event.reward
			self.logged_observations = self.star.define_logged_observations()
			self.logged_observations = self.update_logged_observations(self.logged_observations)
			if log_iterations is True:
				bonsai_tools.log_iteration(self.star.logger, self.logged_observations)
			else:
				bonsai_tools.print_progress(self.logged_observations)

		# The episode is terminal.  Finish the episode. 
		elif isinstance(event, EpisodeFinishEvent):
			log.event("Episode Finish")	
			print("episode count: {},  iteration count: {}, episode reward: {:6.2f}".format(
				self.episode_count, self.iteration_count, self.episode_reward))

		elif isinstance(event, FinishedEvent):
			log.event("Finished")
			return False

		elif event is None:
			return False
		return True

	def start_episode(self, config=None):
		self.star.simulator_reset_config()
		self.reset_iteration_metrics()

	def reset_iteration_metrics(self):
		"""Executed once every start of episode
		"""
		self.reward = 0
		self.terminal = False
		self.episode_reward = 0.0
		self.iteration_count = 0
		self._cpu_pc = psutil.cpu_percent()
		self._vmem = psutil.virtual_memory().percent
		
	def update_logged_observations(self, logged_observations):
		self._cpu_pc = psutil.cpu_percent()
		self._vmem = psutil.virtual_memory().percent
		updated_observations = {
			'episode_count':self.episode_count,
			'iteration_count':self.iteration_count,
			'terminal':self.terminal,
			'reward':self.reward,
			'episode_reward':self.episode_reward,
			'cpu_pc':self._cpu_pc,
			'vmem':self._vmem
		}
		#updated_observations.update(self.action)
		updated_observations.update(logged_observations)
		return updated_observations

def run_brain():
	config = Config(sys.argv)
	brain = Brain(config)
	print("start connect")
	sim = ModelConnector(brain, "the_simulator", config) #this sim name needs to match what is declared in inkling
	if brain.config.predict:
		trainer = ModelTrainer(sim, predict=True)
		print("start predicting")
	else:
		print("start training")
		trainer = ModelTrainer(sim, predict=False)
	while trainer.run():
		pass

if __name__ == "__main__":
	run_brain()