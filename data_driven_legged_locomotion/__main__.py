import numpy as np
from pathlib import Path
import mujoco.viewer as viewer
import time

from data_driven_legged_locomotion.agents.pendulum import SwingUpService, LQRService
from data_driven_legged_locomotion.common import MujocoEnvironment, ServiceSet, MaxEntropyCrowdsouring
from data_driven_legged_locomotion.tasks.pendulum import PendulumEnvironment

env = PendulumEnvironment()
ss = env.ss
model = env.model

def cost(states,k):
  g = 9.81
  m = 5.5
  l = 0.5
  if len(states.shape) == 1:
    states = np.expand_dims(states, axis=0)
  #states is now n_samples x n_states
  #ref = np.array([np.pi, 0])
  #cost = np.sum((states-ref)**2, axis=1)
  E_actual = 0.5 * m * l**2 * states[:,1]**2 - m * g * l * np.cos(states[:,0])
  E_desired = m * g * l
  pos_actual = states[:,0]
  pos_desired = np.pi
  cost = (E_actual - E_desired)**2 + 100*(pos_actual - pos_desired)**2
  print(f"energy error: {np.mean((E_actual - E_desired)**2)}")
  print(f"position error: {np.mean(100*(pos_actual - pos_desired)**2)}")
  cost = np.squeeze(cost)
  return cost

services = ServiceSet(ss)
SwingUpService = SwingUpService(ss, model)
lqrService = LQRService(ss, model)
services.addService(SwingUpService)
services.addService(lqrService)
crowdsourcing = MaxEntropyCrowdsouring(ss, services, cost)

def get_control(env):
  x = env.get_state()
  q, dot_q = env.get_state(split=True)
  x_index = ss.toIndex(x)
  crowdsourcing.initialize(x)
  service_list, behavior = crowdsourcing.run()
  service_index = service_list[0]
  if service_index == 0:
    u = SwingUpService.policy(x)
  elif service_index == 1:
    u = lqrService.policy(x)
  return u

with env.launch_passive_viewer() as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running():
    step_start = time.time()

    #control_callback(model,data)
    # Step the simulation forward.
    #mujoco.mj_step(model, data)
    
    u = get_control(env)
    env.step(u)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = env.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)