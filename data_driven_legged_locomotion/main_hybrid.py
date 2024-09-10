from datetime import datetime
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from pathlib import Path
import logging
import mujoco
import mujoco.viewer as viewer
import time

from data_driven_legged_locomotion.common import ServiceSet, GreedyMaxEntropyCrowdsouring
from data_driven_legged_locomotion.tasks.h1_walk import H1WalkEnvironment, h1_walk_cost
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
model = env.model
cost = h1_walk_cost

# Crowdsourcing
services = ServiceSet(ss)

# Video
video_fps = 60
video_resolution = (720, 1280)
frame_count = 0
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
video_path = Path(__file__).parent.parent / "videos" / f"h1_walk_{current_datetime}.mp4"
if not video_path.parent.exists():
		video_path.parent.mkdir()
renderer = mujoco.Renderer(model, height=video_resolution[0], width=video_resolution[1])

# Services
variances = np.ones(ss.n_states) * 0.000001
hybrid_service = HybridTDMPCService(ss, model, variances=variances, zoh=True)
services.addService(hybrid_service)
hybrid_service_2 = HybridTDMPCService(ss, model, variances=variances, zoh=True)
hybrid_service_2.set_policy_reference(np.array([0.0, 0.0, 0.98, 0.7071068, 0, 0, 0.7071068]))
services.addService(hybrid_service_2)

# Crowdsourcing
crowdsourcing = GreedyMaxEntropyCrowdsouring(ss, services, cost)

# Logging header
log_header = ["Time", "State"]
log_header = log_header + [f"Service_{i}_NextState" for i,_ in enumerate(services.services)]
log_header = log_header + ["Service_Index", "Control"]
logger.log(logging.INFO, log_header)
log_row = []

def get_control(env):
	x = env.get_state()
	log_row.append(list(x))

	# Update the data for the services
	for index,service in enumerate(services.services):
		service.set_data(env.data)

	crowdsourcing.initialize(x, time=env.time)
	for i in range(len(services.services)):
		next_state = crowdsourcing._behaviors.behaviors[i].getAtTime(0).pf.mean
		log_row.append(list(next_state))
	service_list, behavior = crowdsourcing.run()
	service_index = service_list[0]
	print(f"[DEBUG] Service index: {service_index}")
	log_row.append(service_index)

	u = services.services[service_index].last_u
	#return u

	u_traj = services.services[service_index].control_trajectory
	return u_traj

with env.launch_passive_viewer() as viewer:
	with media.VideoWriter(video_path, fps=video_fps, shape=video_resolution) as video:
		# Close the viewer automatically after 30 wall-seconds.
		start = time.time()
		while viewer.is_running():
			log_row = []
			log_row.append(env.time)
			step_start = time.time()

			# Step the simulation forward.
			#u = get_control(env)
			u_traj = get_control(env)
			for u in u_traj:
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
			
			# Rudimentary time keeping, will drift relative to wall clock.
			time_until_next_step = env.timestep - (time.time() - step_start)
			if time_until_next_step > 0:
				time.sleep(time_until_next_step)