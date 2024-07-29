import numpy as np
import mujoco.viewer as viewer
import time

from data_driven_legged_locomotion.common import ServiceSet, GreedyMaxEntropyCrowdsouring
from data_driven_legged_locomotion.tasks.h1_walk import H1WalkEnvironment, h1_walk_cost

from data_driven_legged_locomotion.agents.tdmpc_service import TDMPCService

from data_driven_legged_locomotion.common import StateSpace
import time
import mujoco
from datetime import datetime
from pathlib import Path
import mediapy as media

#env = PendulumEnvironment()
env = H1WalkEnvironment()
ss = env.ss
model = env.model
cost = h1_walk_cost

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


mujoco_tdmpc_service = TDMPCService(ss, model)
mujoco_tdmpc_service_2 = TDMPCService(ss, model)
mujoco_tdmpc_service_2.set_policy_reference(np.array([0.0, 0.0, 0.98, 0.7071068, 0, 0, 0.7071068]))
services.addService(mujoco_tdmpc_service)
services.addService(mujoco_tdmpc_service_2)

crowdsourcing = GreedyMaxEntropyCrowdsouring(ss, services, cost)

def get_control(env):
  x = env.get_state()
  q, dot_q = env.get_state(split=True)
  x_index = ss.toIndex(x)
  
  crowdsourcing.initialize(x)
  service_list, behavior = crowdsourcing.run()
  service_index = service_list[0]
  print("Service index: ", service_index)
  u = services.services[service_index].last_u

  return u

with env.launch_passive_viewer() as viewer:
  with media.VideoWriter(video_path, fps=video_fps, shape=video_resolution) as video:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running():
      #print(f"[DEBUG] Iteration start")
      step_start = time.time()

      # Step the simulation forward.
      #control_time = time.time()
      u = get_control(env)
      #control_time = time.time() - control_time
      #print(f"[DEBUG] Total control time: {control_time}")
      #env_setp_time = time.time()
      env.step(u)
      #env_setp_time = time.time() - env_setp_time
      #print(f"[DEBUG] Environment step time: {env_setp_time}")

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