
# misc shared tools
import argparse, logging, datetime, time, os, io, copy, json, subprocess
import numpy as np
from threading import Thread
from collections import deque

def log_initialize( brain, pathname='./log', log_training_speed = False):
	#initialize logfile to collect results
	directory = os.path.dirname(pathname) # check if pathname exists otherwise create it 
	if not os.path.exists(directory):os.makedirs(directory)
	brainname = brain.name
	predictflag = brain.config.predict
	
	
	if log_training_speed == True:
		filenameSuffix = brainname + '_training_speed_monitoring.csv'
	else:
		if predictflag == True:
			filenameSuffix = brainname + '_PREDICT.csv'
		else:
			filenameSuffix = brainname + '_TRAINING.csv'
	return log_create(pathname, filenameSuffix )

def log_create( pathname, nameIt, timestamp_flag = False):
	directory = os.path.dirname(pathname) # check if pathname exists otherwise create it 
	if not os.path.exists(directory):
		os.makedirs(directory)
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)
	timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
	filename = pathname  + timestamp + '_' + nameIt
	print( "created logging file: " + filename)
	handler = logging.FileHandler(filename)
	handler.setLevel(logging.INFO)
	if timestamp_flag ==True:
		formatter = logging.Formatter('%(asctime)s, %(message)s')  # create a logging format
	else:
		formatter = logging.Formatter('%(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)  # add the handlers to the logger
	return logger

def log_observations_columns(logger, logged_observations_dict):
	column_names_string = ''
	for item in logged_observations_dict.keys():
		column_names_string = column_names_string + item + ', ' 
	logger.info(column_names_string[0:-2])

def log_iteration(logger, logged_observations_dict):
	log_string = ''
	for item in logged_observations_dict.keys():
		log_string = log_string + '{'+str(item)+'}' + ', ' 
	logger.info(log_string[0:-2].format(**logged_observations_dict))

def print_progress(logged_observations_dict, n=5):
	if logged_observations_dict['iteration_count'] % n ==0 : 
		print('episode count: {}, iteration count {}, iteration reward: {:6.3f}'.format(
			logged_observations_dict['episode_count'], 
			logged_observations_dict['iteration_count'],
			logged_observations_dict['reward']), end ='\r')
	else:
		pass

def rename_action(dictionary,prefix='none'):
	new_dictionary = copy.deepcopy(dictionary)
	for old_key in dictionary:
		new_key = prefix + str(old_key)
		new_dictionary[new_key] = new_dictionary.pop(old_key)
	return new_dictionary

def normalize(value,mean,variance):
	normalized_value = (value-mean)/np.abs(variance)
	return normalized_value

#----------------------------------------
# state history and state transformations
#----------------------------------------

def augment_state_with_history(state, history_queue, iteration_history = 1):
	history_queue.appendleft(state)
	state_with_history = copy.deepcopy(state)
	for iteration in range(1,iteration_history+1):
		for key in state:
			new_key = str(key)+str(iteration)
			state_with_history[new_key] = history_queue[iteration][key]
	return [state_with_history, history_queue]

def initialize_history_queue(state, iteration_history = 1):
	initial_history = [state]
	for iteration in range(0,iteration_history+1):
		initial_history.append(state)
		state_history = deque(initial_history,maxlen=iteration_history+1)
	return state_history

#-------------------------
# Training speed monitoring
#-------------------------

def get_training_status(brainname):
	"""returns a dictionary of current training status (iterations, concept, etc.)
	"""
	cp = subprocess.run('bonsai train status --json --brain {}'.format(brainname),
				stderr=subprocess.PIPE,
				stdout=subprocess.PIPE,
				shell=True)
	status = json.loads(cp.stdout)	
	return status

def get_iteration_rate(brainname, time_interval=10):
	"""retruns the current iteration rate in [number of iterations / s]. time_interval in seconds
	"""
	status = get_training_status(brainname)
	num_iterations_before = status['iteration']
	time.sleep(time_interval)
	status = get_training_status(brainname)
	num_iterations_after = status['iteration']
	iteration_rate = (num_iterations_after - num_iterations_before)/time_interval
	return [iteration_rate,status['iteration']]

def get_num_of_sims(brainname):
	"""returns number of active sims 
	"""
	cp = subprocess.run('bonsai sims list --json --brain {}'.format(brainname),
				stderr=subprocess.PIPE,
				stdout=subprocess.PIPE,
				shell=True)
	sims_status = json.loads(cp.stdout)
	keys = list(sims_status.keys())
	sim_list = sims_status[keys[0]]['active']
	return len(sim_list)

def thread_monitor_training(monitoring_logger,logged_observations_dict, brainname):
	while True:
		logged_observations_dict['num_of_sims'] = get_num_of_sims(brainname)
		[logged_observations_dict['num_iterations_per_s'],logged_observations_dict['iterations']] = get_iteration_rate(brainname)
		logged_observations_dict['datetime']=datetime.datetime.now() 
		log_iteration(monitoring_logger, logged_observations_dict)

def monitor_training(monitoring_logger,logged_observations_dict, brainname):
	thread = Thread(target = thread_monitor_training, args = (monitoring_logger,logged_observations_dict, brainname))
	thread.start()

if __name__ == "__main__":
	pass