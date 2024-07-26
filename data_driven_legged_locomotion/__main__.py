import numpy as np
from pathlib import Path
import mujoco.viewer as viewer
import time

from data_driven_legged_locomotion.agents.h1_walk import MujocoMPCService
from data_driven_legged_locomotion.common import MujocoEnvironment, ServiceSet, GreedyMaxEntropyCrowdsouring
from data_driven_legged_locomotion.tasks.h1_walk import H1WalkEnvironment, h1_walk_cost

#env = PendulumEnvironment()
env = H1WalkEnvironment()
ss = env.ss
model = env.model
cost = h1_walk_cost

services = ServiceSet(ss)
# swingUpService = SwingUpService(ss, model)
# lqrService = LQRService(ss, model)
# services.addService(swingUpService)
# services.addService(lqrService)
mujoco_mpc_service_forward = MujocoMPCService(ss, model, direction=np.array([10.0, 0.0]))
mujoco_mpc_service_right = MujocoMPCService(ss, model, direction=np.array([0.0, 10.0]))
services.addService(mujoco_mpc_service_forward)
services.addService(mujoco_mpc_service_right)
crowdsourcing = GreedyMaxEntropyCrowdsouring(ss, services, cost)

def get_control(env):
  x = env.get_state()
  q, dot_q = env.get_state(split=True)
  x_index = ss.toIndex(x)
  init_time = time.time()
  crowdsourcing.initialize(x, time=env.time)
  init_time = time.time() - init_time
  print(f"[DEBUG] Initialization time: {init_time}")
  crowdsourcing_time = time.time()
  service_list, behavior = crowdsourcing.run()
  crowdsourcing_time = time.time() - crowdsourcing_time
  print(f"[DEBUG] Total crowdsourcing time: {crowdsourcing_time}")
  service_index = service_list[0]
  u = services.services[service_index].last_u
  return u

with env.launch_passive_viewer() as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running():
    print(f"[DEBUG] Iteration start")
    step_start = time.time()

    # Step the simulation forward.
    control_time = time.time()
    u = get_control(env)
    control_time = time.time() - control_time
    print(f"[DEBUG] Total control time: {control_time}")
    env_setp_time = time.time()
    env.step(u)
    env_setp_time = time.time() - env_setp_time
    print(f"[DEBUG] Environment step time: {env_setp_time}")

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = env.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)