import numpy as np
from pathlib import Path
import mujoco.viewer as viewer
import time

from data_driven_legged_locomotion.agents.h1_walk import MujocoMPCService
from data_driven_legged_locomotion.common import MujocoEnvironment, ServiceSet, GreedyMaxEntropyCrowdsouring
from data_driven_legged_locomotion.tasks.h1_walk import H1WalkEnvironment, h1_walk_cost

from data_driven_legged_locomotion.agents.tdmpc_service import TDMPCService
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


#mujoco_mpc_service = MujocoMPCService(ss, model)
#services.addService(mujoco_mpc_service)

mujoco_tdmpc_service = TDMPCService(ss, model, agent_path="/home/davide/data-driven-legged-locomotion/data_driven_legged_locomotion/agents/tdmpc/config/step-660791.pt", config_path="/home/davide/data-driven-legged-locomotion/data_driven_legged_locomotion/agents/tdmpc/config/config.yaml")
services.addService(mujoco_tdmpc_service)

crowdsourcing = GreedyMaxEntropyCrowdsouring(ss, services, cost)

def get_control(env):
  x = env.get_state()
  q, dot_q = env.get_state(split=True)
  x_index = ss.toIndex(x)
  crowdsourcing.initialize(x)
  service_list, behavior = crowdsourcing.run()
  service_index = service_list[0]
  u = services.services[service_index].last_u
  return u

with env.launch_passive_viewer() as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running():
    step_start = time.time()

    # Step the simulation forward.
    u = get_control(env)
    env.step(u)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = env.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)