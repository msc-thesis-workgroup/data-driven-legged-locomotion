from datetime import datetime
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import pathlib
import logging
import mujoco
import mujoco.viewer as viewer
import time

import data_driven_legged_locomotion.agents.h1_walk.mujoco_mpc.services as mujoco_mpc_agent_module
from data_driven_legged_locomotion.agents.h1_walk.mujoco_mpc import H1PolarWalkAgent
from data_driven_legged_locomotion.agents.h1_walk import FNOStateSpace, OfflineAR2Service
from data_driven_legged_locomotion.agents.h1_walk import MujocoMPCServiceV2 as MujocoMPCService
from data_driven_legged_locomotion.common import MujocoEnvironment, ServiceSet, GreedyMaxEntropyCrowdsouring
from data_driven_legged_locomotion.tasks.h1_walk import H1WalkEnvironment, H1WalkMapEnvironment, h1_quadratic_objective as h1_walk_cost
from data_driven_legged_locomotion.utils import CsvFormatter

from data_driven_legged_locomotion.agents.tdmpc_service import TDMPCService

# Experiment info+
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_name = f"h1_walk_{current_datetime}"
experiment_folder = pathlib.Path(__file__).parent.parent / "experiments" / experiment_name
if not experiment_folder.exists():
    experiment_folder.mkdir(parents=True)

# Logging
logger = logging.getLogger(__name__)
fileHandler = logging.FileHandler(experiment_folder / "experiment_data.csv")
fileHandler.setFormatter(CsvFormatter())
logger.addHandler(fileHandler)

XX1 = [[],[]]
XX2 = [[],[]]
PP = []

# Environment
#env = PendulumEnvironment()
env = H1WalkEnvironment()
ss = env.ss
model = env.model
cost = h1_walk_cost

# Video
video_fps = 60
video_resolution = (720, 1280)
frame_count = 0

video_path = experiment_folder / f"h1_walk_{current_datetime}.mp4"
if not video_path.parent.exists():
    video_path.parent.mkdir()
renderer = mujoco.Renderer(model, height=video_resolution[0], width=video_resolution[1])

# Agent
agent = H1PolarWalkAgent(env)

# Crowdsourcing
services = ServiceSet(ss)
ss = FNOStateSpace()
current_dir = pathlib.Path(mujoco_mpc_agent_module.__file__).parent
models_file = current_dir / "models_ols_fno.pkl"
service_forward = OfflineAR2Service(ss, models_file, "FORWARD")
service_left = OfflineAR2Service(ss, models_file, "LEFT")
service_right = OfflineAR2Service(ss, models_file, "RIGHT")
crowdsourcing = GreedyMaxEntropyCrowdsouring(ss, services, cost)

#fno
last_fno = np.zeros((2,3))
qpos_old = env.data.qpos[:7][:]
model_timestep = 0.02
downsample_factor = int(model_timestep / 0.002)
i = 0

# swingUpService = SwingUpService(ss, model)
# lqrService = LQRService(ss, model)
# services.addService(swingUpService)
# services.addService(lqrService)
#mujoco_mpc_service_forward = MujocoMPCService(ss, model, direction=np.array([10.0, 0.0]), env=env)
#mujoco_mpc_service_right = MujocoMPCService(ss, model, direction=np.array([0.0, 10.0]), env=env)
# tdmpc_service_1 = TDMPCService(ss, model)
# tdmpc_service_2 = TDMPCService(ss, model)
# tdmpc_service_2.set_policy_reference(np.array([0.0, 0.0, 0.98, 0.7071068, 0, 0, 0.7071068]))
#services.addService(mujoco_mpc_service_forward)
#services.addService(mujoco_mpc_service_right)
# services.addService(tdmpc_service_1)
# services.addService(tdmpc_service_2)

log_header = ["Time", "State"]
log_header = log_header + [f"Service_{i}_NextState" for i,_ in enumerate(services.services)]
log_header = log_header + ["Service_Index", "Control"]
logger.log(logging.INFO, log_header)
log_row = []

def get_control(env):
  x = env.get_state()
  log_row.append(list(x))
  q, dot_q = env.get_state(split=True)
  x_index = ss.toIndex(x)
  init_time = time.time()
  crowdsourcing.initialize(x, time=env.time)
  for i in range(2):
    next_state = crowdsourcing._behaviors.behaviors[i].getAtTime(0).pf.mean
    log_row.append(list(next_state))
    XX1[i].append(next_state[0])
    XX2[i].append(next_state[1])
    if len(XX1[i]) > 100:
      XX1[i].pop(0)
      XX2[i].pop(0)
  init_time = time.time() - init_time
  print(f"[DEBUG] Initialization time: {init_time}")
  crowdsourcing_time = time.time()
  service_list, behavior = crowdsourcing.run()
  crowdsourcing_time = time.time() - crowdsourcing_time
  print(f"[DEBUG] Total crowdsourcing time: {crowdsourcing_time}")
  service_index = service_list[0]
  log_row.append(service_index)
  PP.append(service_index)
  if len(PP) > 100:
    PP.pop(0)
  u = services.services[service_index].last_u
  return u

def get_control2(env):
  crowdsourcing.initialize(last_fno, time=env.time)
  service_list, behavior = crowdsourcing.run()
  service_index = service_list[0]
  if service_index == 0:
    agent.set_command("FORWARD")
  elif service_index == 1:
    agent.set_command("LEFT")
  elif service_index == 2:
    agent.set_command("RIGHT")
  u = agent.get_control_action()
  return u
  

def plot_sequence():
  plt.figure(1)
  plt.clf()
  plt.stem(PP)
  plt.pause(0.0001)
  plt.figure(2)
  plt.clf()
  ax = plt.gca()
  sigma = 0.01
  plt.scatter(XX1[0], XX2[0], c='r')
  plt.scatter(XX1[1], XX2[1], c='b')
  for i in range(len(XX1[0])):
    circle1 = plt.Circle((XX1[0][i], XX2[0][i]), sigma*2, color='r', fill=False)
    circle2 = plt.Circle((XX1[1][i], XX2[1][i]), sigma*2, color='b', fill=False)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
  plt.pause(0.0001)


with env.launch_passive_viewer() as viewer:
  with media.VideoWriter(video_path, fps=video_fps, shape=video_resolution) as video:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running():
      log_row = []
      log_row.append(env.time)
      step_start = time.time()

      # Step the simulation forward.
      control_time = time.time()
      u = get_control(env)
      log_row.append(list(u))
      control_time = time.time() - control_time
      env.step(u)

      # Pick up changes to the physics state, apply perturbations, update options from GUI.
      viewer.sync()
      
      # Update fno
      if i % downsample_factor == 0:
          last_fno[1] = last_fno[0]
          last_fno[0] = H1WalkEnvironment.get_fno_from_delta(env.data.qpos, qpos_old, model_timestep)
          qpos_old = env.data.qpos[:7][:]
      
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