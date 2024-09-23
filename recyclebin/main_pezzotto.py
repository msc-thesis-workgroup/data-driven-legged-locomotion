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
from data_driven_legged_locomotion.tasks.h1_walk import H1WalkEnvironment, h1_walk_cost, h1_walk_cost_trajectory
from data_driven_legged_locomotion.utils import CsvFormatter

from recyclebin.tdmpc_service import TDMPCService, TDMPCServiceV2


AGENT_HORIZON = 100 # 500*0.002 = 1s


# Experiment info
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_name = f"h1_walk_{current_datetime}"
experiment_folder = Path(__file__).parent.parent / "experiments" / experiment_name
if not experiment_folder.exists():
    experiment_folder.mkdir(parents=True)


# create csv file to store data

with open(experiment_folder / "experiment_data.csv", "w") as file:
  file.write("time,service_index\n")


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


# mujoco_tdmpc_service = TDMPCServiceV2(ss, model, agent_horizon=AGENT_HORIZON)
mujoco_tdmpc_service_2 = TDMPCServiceV2(ss, model, agent_horizon=AGENT_HORIZON)
mujoco_tdmpc_service_2.set_policy_reference(np.array([0.0, 0.0, 0.98, 0.7071068, 0, 0, 0.7071068]))
# services.addService(mujoco_tdmpc_service)
services.addService(mujoco_tdmpc_service_2)

crowdsourcing = GreedyMaxEntropyCrowdsouring(ss, services, cost)


def get_control(env):
  x = env.get_state()

  #crowdsourcing.initialize(x, time=env.time)
  #service_list, behavior = crowdsourcing.run()
  #service_index = service_list[0]
  
  state_trajectory = []
  costs = []

  # Get the next state for each service
  for index,service in enumerate(services.services):
    service.set_data(env.data)
    state_trajectory.append(service._get_next_state(x,env.time))
    
    # workaround to get the cost of the service without using the crowdsourcing algorithm
    costs.append(h1_walk_cost_trajectory(state_trajectory[index],env.time))
    print(f"[DEBUG] Service {index} cost: {costs[index]} ")
  
  service_index = np.argmin(costs)

  # Set the agent of the selected service to the other services to synchronize the agents
  agent = services.services[service_index].get_agent_copy()
  for index,service in enumerate(services.services):
    if index != service_index:
      service.set_agent_copy(agent)
  
  print(f"[DEBUG] Service index: {service_index}")

  with open(experiment_folder / "experiment_data.csv", "a") as file:
    file.write(f"{env.time},{service_index}\n")

  u = services.services[service_index].u_trajectory
  #u = services.services[service_index].last_u
  return u

with env.launch_passive_viewer() as viewer:
  with media.VideoWriter(video_path, fps=video_fps, shape=video_resolution) as video:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running():
      #print(f"[DEBUG] Iteration start")
      step_start = time.time()
      
      # Step the simulation forward.
      control_trajectory = get_control(env)
      for u in control_trajectory:
        env.step(u)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
        
        # Render video
        if frame_count < env.time * video_fps:
            renderer.update_scene(env.data, camera="top")
            pixels = renderer.render()
            video.add_image(pixels)
            frame_count += 1
    
        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = env.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)