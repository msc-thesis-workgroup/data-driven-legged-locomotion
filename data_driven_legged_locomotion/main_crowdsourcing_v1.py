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

from data_driven_legged_locomotion.agents.tdmpc_service import TDMPCService, TDMPCServiceV2

# Experiment info
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_name = f"h1_walk_{current_datetime}"
experiment_folder = Path(__file__).parent.parent / "experiments" / experiment_name
if not experiment_folder.exists():
    experiment_folder.mkdir(parents=True)


# Environment
env = H1WalkEnvironment()
value = 1000000

# set actuator control range: -value to value for all actuators
env.model.actuator_ctrlrange = np.array([[-value, value]] * env.model.nu)

print(f"[DEBUG] Actuator control range: {env.model.actuator_ctrlrange}")

ss = env.ss
model = env.model
cost = h1_walk_cost

# Video
video_fps = 60
video_resolution = (720, 1280)
frame_count = 0
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
video_path = Path(__file__).parent.parent / "videos" / f"h1_walk_{current_datetime}.mp4"
if not video_path.parent.exists():
    video_path.parent.mkdir()
renderer = mujoco.Renderer(model, height=video_resolution[0], width=video_resolution[1])

# Crowdsourcing
services = ServiceSet(ss)

AGENT_HORIZON = 50 # 100*0.002 = 0.2s
variances = np.ones(ss.n_states) * 0.000001
mujoco_tdmpc_service = TDMPCServiceV2(ss, model, agent_horizon=AGENT_HORIZON, variances=variances)
mujoco_tdmpc_service_2 = TDMPCServiceV2(ss, model, agent_horizon=AGENT_HORIZON, variances=variances)
mujoco_tdmpc_service_2.set_policy_reference(np.array([0.0, 0.0, 0.98, 0.7071068, 0, 0, 0.7071068]))
services.addService(mujoco_tdmpc_service)
services.addService(mujoco_tdmpc_service_2)

crowdsourcing = GreedyMaxEntropyCrowdsouring(ss, services, cost)



def get_control(env):
  x = env.get_state()
  
  q, dot_q = env.get_state(split=True)
  x_index = ss.toIndex(x)
  init_time = time.time()

  for index,service in enumerate(services.services):
    service.set_data(env.data)

  crowdsourcing.initialize(x, time=env.time)
   
  service_list, behavior = crowdsourcing.run()
  service_index = service_list[0]

  # Only for TD-MPC2 service. TODO Refactor code to make it more general
  # Set the agent of the selected service to the other services to synchronize the agents
  agent = services.services[service_index].get_agent_copy()
  for index,service in enumerate(services.services):
    if index != service_index:
      service.set_agent_copy(agent)
  
  print(f"[DEBUG] Service index: {service_index}")


  desired_state = behavior.getAtTime(0).pf.mean
  return desired_state

with env.launch_passive_viewer() as viewer:
  with media.VideoWriter(video_path, fps=video_fps, shape=video_resolution) as video:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running():
      #print(f"[DEBUG] Iteration start")
      
      step_start = time.time()

      #u = get_control(env)
      #env.step(u)
      desired_state = get_control(env)
      env.reach_state(desired_state, AGENT_HORIZON,viewer=viewer)


      # Pick up changes to the physics state, apply perturbations, update options from GUI.
      viewer.sync()
      
      # Render video
      if frame_count < env.time * video_fps:
          renderer.update_scene(env.data, camera="top")
          pixels = renderer.render()
          video.add_image(pixels)
          frame_count += 1

      # Log the data
      
      
      # Rudimentary time keeping, will drift relative to wall clock.
      time_until_next_step = env.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)