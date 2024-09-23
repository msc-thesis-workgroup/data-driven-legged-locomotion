from datetime import datetime
import mediapy as media
import numpy as np
from pathlib import Path
import logging
import mujoco
import mujoco.viewer as viewer
import time

from data_driven_legged_locomotion.common import ServiceSet, GreedyMaxEntropyCrowdsouring
from data_driven_legged_locomotion.tasks.h1_walk import H1WalkEnvironment, Cost
from data_driven_legged_locomotion.utils import CsvFormatter

from data_driven_legged_locomotion.agents.HybridTDMPC_service import HybridTDMPCService

# Experiment info+
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_name = f"h1_walk_{current_datetime}"
experiment_folder = Path(__file__).parent.parent / "experiments" / experiment_name
if not experiment_folder.exists():
		experiment_folder.mkdir(parents=True)

# Logging
logger = logging.getLogger(__name__)
fileHandler = logging.FileHandler(experiment_folder / "experiment_data.csv")
fileHandler.setFormatter(CsvFormatter())
logger.addHandler(fileHandler)

# Environment
env = H1WalkEnvironment()
ss = env.ss

# Define obstacles positions and sizes
obstacle_positions = [
	[3.0, 3.0, 0],  # Position of the first obstacle
    [4.0, 4.0, 0],  # Position of the second obstacle
	[5.0, 5.0, 0],  # Position of the third obstacle
	[6.0, 6.0, 0],  # Position of the fourth obstacle
]
obstacle_sizes = [
	[0.7, 0.7, 3],  # Size of the first obstacle
    [0.7, 0.7, 4],  # Size of the second obstacle
	[0.7, 0.7, 5],  # Size of the third obstacle
	[0.7, 0.7, 6],  # Size of the fourth obstacle
]
obstacle_rgba = [
    [0, 1, 0, 1],  # Green for the first obstacle
	[0, 0, 1, 1],  # Blue for the second obstacle
	[1, 1, 0, 1],  # Yellow for the third obstacle
	[1, 0, 1, 1],  # Magenta for the fourth obstacle
]

# Create obstacles
env.create_obstacles(obstacle_positions, obstacle_sizes, obstacle_rgba)
model = env.model

# Cost
cost_obj = Cost(obstacle_positions, obstacle_sizes)
cost = cost_obj.get_cost_function()

# Crowdsourcing
services = ServiceSet(ss)

# Video parameters
video_fps = 60
video_resolution = (720, 1280)
frame_count = 0
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
video_path = Path(__file__).parent.parent / "videos" / f"h1_walk_{current_datetime}.mp4"
if not video_path.parent.exists():
		video_path.parent.mkdir()
renderer = mujoco.Renderer(model, height=video_resolution[0], width=video_resolution[1])

# Defining Services
variances = np.ones(ss.n_states) * 0.000001

FRAME_SKIP = 1
AGENT_HORIZON = 1
N_SAMPLES = 500
DELTA_STEP = 0.003
hybrid_service = HybridTDMPCService(ss, model, variances=variances, agent_horizon=AGENT_HORIZON, frame_skip=FRAME_SKIP, delta_step=DELTA_STEP)
services.addService(hybrid_service)
hybrid_service_2 = HybridTDMPCService(ss, model, variances=variances, agent_horizon=AGENT_HORIZON, frame_skip=FRAME_SKIP, delta_step=DELTA_STEP)
hybrid_service_2.set_policy_reference(np.array([0.9238795, 0, 0, 0.3826834]))
services.addService(hybrid_service_2)
hybrid_service_3 = HybridTDMPCService(ss, model, variances=variances, agent_horizon=AGENT_HORIZON, frame_skip=FRAME_SKIP, delta_step=DELTA_STEP)
hybrid_service_3.set_policy_reference(np.array([0.7071068, 0, 0, 0.7071068]))
services.addService(hybrid_service_3)

# Crowdsourcing
crowdsourcing = GreedyMaxEntropyCrowdsouring(ss, services, cost, n_samples=N_SAMPLES)

# Logging header
log_header = ["Time", "State"]
log_header = log_header + [f"Service_{i}_NextState" for i,_ in enumerate(services.services)]
log_header = log_header + ["Service_Index", "Control"]
logger.log(logging.INFO, log_header)
log_row = []



def get_control(env):

	x = env.get_state().copy()
	log_row.append(list(x))

	# Update the data for the services
	for index,service in enumerate(services.services):
		#print("index",index ,"id: ", id(service))
		service.set_data(env.data)

	crowdsourcing.initialize(x, time=env.time)

	# Getting information from the crowdsourcing for the log
	for i in range(len(services.services)):
		next_state = crowdsourcing._behaviors.behaviors[i].getAtTime(0).pf.mean
		#print("service: ", i,"next_state: ", next_state[0:3])
		log_row.append(list(next_state))
	
	# Running the crowdsourcing
	service_list, behavior = crowdsourcing.run()
	service_index = service_list[0]
	
	print(f"[DEBUG] Chosen Service index: {service_index}")
	log_row.append(service_index)
	
	u = services.services[service_index].get_control(x, t=env.time)
	return u


first_time = True
old_agent_index = None
service_index = None
def get_control_without_crowdsourcing(env):
	global first_time
	global old_agent_index
	global service_index

	x = env.get_state()
	log_row.append(list(x))

	# Update the data for the services
	for index,service in enumerate(services.services):
		#print("index",index ,"id: ", id(service))
		service.set_data(env.data)

	# Sol TESTING ONLY
	costs = []
	for index,service in enumerate(services.services):

		state = x.copy()
		if index == 0:
			state[0] += 0.003
		elif index == 1:
			state[0] += 0.003/np.sqrt(2)
			state[1] += 0.003/np.sqrt(2)
		elif index == 2:
			state[1] += 0.003
		log_row.append(list(state))
		costs.append(cost(state, 0))
				
	service_index = np.argmin(costs)
	
	print(f"[DEBUG] Service index: {service_index}")
	log_row.append(service_index)


	# Cheating block
	if not first_time and old_agent_index != service_index:
		agent = services.services[old_agent_index].get_agent_copy()
		services.services[service_index].set_agent_copy(agent)

		#env.data.qvel[3:6] = env.data.qvel[3:6] * 0.5
		env.data.qvel[0:3] = env.data.qvel[0:3] * 0.33
		services.services[service_index].set_data(env.data)

	if first_time:
		first_time = False

	old_agent_index = service_index

	winner_service = services.services[service_index]
	u_traj = winner_service.get_control_trajectory(x, t=env.time) # t is the same time of .initialize(x, time=env.time)
	
	return u_traj[-1] # the trajectory is made by only one element. TODO REFACTOR THIS TO BE MORE CLEAR

def get_next_state(env):
	x = env.get_state()
	#log_row.append(list(x))

	# Update the data for the services
	for index,service in enumerate(services.services):
		service.set_data(env.data)

	crowdsourcing.initialize(x, time=env.time)
	for i in range(len(services.services)):
		next_state = crowdsourcing._behaviors.behaviors[i].getAtTime(0).pf.mean
		#log_row.append(list(next_state))
	service_list, behavior = crowdsourcing.run()
	service_index = service_list[0]
	print(f"[DEBUG] Service index: {service_index}")
	#log_row.append(service_index)

	#u = services.services[service_index].last_u
	#return u

	# agent = services.services[service_index].get_agent_copy()
	# for index,service in enumerate(services.services):
	# 	if index != service_index:
	# 		service.set_agent_copy(agent)

	desired_state = crowdsourcing._behaviors.behaviors[service_index].getAtTime(0).pf.mean
	return desired_state


with env.launch_passive_viewer() as viewer:
	with media.VideoWriter(video_path, fps=video_fps, shape=video_resolution) as video:
		# Close the viewer automatically after 30 wall-seconds.
		start = time.time()
		while viewer.is_running():
			log_row = []
			log_row.append(env.time)
			step_start = time.time()

			# Step the simulation forward.
			
			# Sol 1
			temp = old_agent_index
			u = get_control(env)
			#u = get_control_without_crowdsourcing(env)
			log_row.append(list(u))
			env.step(u)

			# Pick up changes to the physics state, apply perturbations, update options from GUI.
			viewer.sync()
			# Render video
			if frame_count < env.time * video_fps:
				renderer.update_scene(env.data, camera="top")
				pixels = renderer.render()
				video.add_image(pixels)
				frame_count += 1
					
			# Log the data
			logger.log(logging.DEBUG, log_row)

			# Sol 2
			# x_state = get_next_state(env)
			# frame_count = env.reach_state(x_state, FRAME_SKIP*AGENT_HORIZON, viewer, video, renderer, frame_count, video_fps)
			# env.data.qfrc_applied = np.zeros_like(env.data.qfrc_applied)
