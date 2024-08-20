from mujoco_mpc import agent as agent_lib
import numpy as np
import pathlib
import time

from data_driven_legged_locomotion.common import MujocoService, StateSpace, Service

class MujocoMPCService(MujocoService):
    def __init__(self, ss: StateSpace, model, variances: float = None, direction: np.ndarray = np.array([1.0, 0.0]), policy_sampling_time: float = 0.02, env = None):
        # We disable zero-order hold as we want the agent to interpolate the control action between the policy sampling time
        super().__init__(ss, model, variances, enable_zoh=False, policy_sampling_time=policy_sampling_time)
        # Find the agent_server binary
        base_path = pathlib.Path(__file__).parent.parent.parent.parent.parent
        print("[DEBUG] base_path: ", base_path)
        server_binary_candidates = [path for path in pathlib.Path(base_path).rglob("agent_server")]
        print("[DEBUG] server_binary_candidates: ", server_binary_candidates)
        if len(server_binary_candidates) == 0:
            raise ValueError(f"Could not find agent_server binary in folder {base_path}, make sure to build the agent_server")
        print("[DEBUG] Agent server binary found: ", server_binary_candidates[0])
        server_binary_path = server_binary_candidates[0]
        # Create and configure the agent
        agent = agent_lib.Agent(task_id="H1 Walk", 
                                model=model, 
                                server_binary_path=server_binary_path)
        #agent.set_cost_weights({"Leg cross": 0.0, "Feet Distance": 0.0, "Face Forward": 1.0})
        agent.set_task_parameter("Torso", 1.3)
        agent.set_task_parameter("Speed", 0.7)
        task_params = agent.get_task_parameters()
        cost_weights = agent.get_cost_weights()
        print("[DEBUG] Task parameters: ", task_params)
        print("[DEBUG] Cost weights: ", cost_weights)
        # Initialize the attributes
        self.agent = agent
        self.direction = direction
        self._last_planning_time = 0.0
        # Update the mocap position
        self._update_mocap_pos()
        self.env = env
        self._sync_env_state()
    
    def _update_mocap_pos(self):
        current_pos = self.data.qpos[:3]
        delta_pos = np.concatenate([self.direction, np.zeros(1)])
        self.data.mocap_pos[0] = current_pos + delta_pos
    
    def _sync_env_state(self):
        self.data.time = self.env.data.time
        self.data.act = self.env.data.act
        self.data.userdata = self.env.data.userdata
    
    def _policy(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        self._update_mocap_pos()
        self._sync_env_state()
        self.agent.set_state(
            time=self.data.time,
            qpos=self.data.qpos,
            qvel=self.data.qvel,
            act=self.data.act,
            mocap_pos=self.data.mocap_pos,
            mocap_quat=self.data.mocap_quat,
            userdata=self.data.userdata,
        )
        if t - self._last_planning_time >= self._policy_sampling_time:
            planner_step_time = time.time()
            self.agent.planner_step()
            planner_step_time = time.time() - planner_step_time
            print(f"[DEBUG] MujocoMPCService {id(self)} planning at time {t}")
            print(f"[DEBUG] MujocoMPCService {id(self)} planner step time: {planner_step_time}")
            self._last_planning_time = t
        u = self.agent.get_action(nominal_action=False, time=t)
        return u

class MujocoMPCServiceV2(MujocoMPCService):
    def _get_next_state(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        self.data.qpos = state[0:self.model.nq]
        self.data.qvel = state[self.model.nq:]
        self._last_u = self._policy(state, t)
        best_trajectory = self.agent.best_trajectory()
        state_traj = best_trajectory["states"]
        return state_traj[-1,:self.ss.n_states]

# class MujocoMPCServiceV2(Service):
#     def __init__(self, model, ss: StateSpace):
#         super().__init__(ss)
#         # Find the agent_server binary
#         base_path = pathlib.Path(__file__).parent.parent.parent.parent.parent
#         print("[DEBUG] base_path: ", base_path)
#         server_binary_candidates = [path for path in pathlib.Path(base_path).rglob("agent_server")]
#         print("[DEBUG] server_binary_candidates: ", server_binary_candidates)
#         if len(server_binary_candidates) == 0:
#             raise ValueError(f"Could not find agent_server binary in folder {base_path}, make sure to build the agent_server")
#         print("[DEBUG] Agent server binary found: ", server_binary_candidates[0])
#         server_binary_path = server_binary_candidates[0]
#         # Create and configure the agent
#         agent = agent_lib.Agent(task_id="H1 Walk", 
#                                 model=model, 
#                                 server_binary_path=server_binary_path)