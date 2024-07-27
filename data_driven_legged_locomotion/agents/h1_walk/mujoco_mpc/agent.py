from mujoco_mpc import agent as agent_lib
import numpy as np
import pathlib
import time

from data_driven_legged_locomotion.common import MujocoService, StateSpace

class MujocoMPCService(MujocoService):
    def __init__(self, ss: StateSpace, model, variances: float = None, direction: np.ndarray = np.array([1.0, 0.0])):
        super().__init__(ss, model, variances, policy_sampling_time=0.02)
        base_path = pathlib.Path(__file__).parent.parent.parent.parent.parent
        print("[DEBUG] base_path: ", base_path)
        server_binary_candidates = [path for path in pathlib.Path(base_path).rglob("agent_server")]
        print("[DEBUG] server_binary_candidates: ", server_binary_candidates)
        if len(server_binary_candidates) == 0:
            raise ValueError(f"Could not find agent_server binary in folder {base_path}, make sure to build the agent_server")
        print("Agent server binary found: ", server_binary_candidates[0])
        server_binary_path = server_binary_candidates[0]
        agent = agent_lib.Agent(task_id="H1 Walk", 
                            model=model, 
                            server_binary_path=server_binary_path)
        agent.set_task_parameter("Torso", 1.3)
        agent.set_task_parameter("Speed", 0.6)
        self.agent = agent
        self.direction = direction
        self._update_mocap_pos()
    
    def _update_mocap_pos(self):
        current_pos = self.data.qpos[:3]
        delta_pos = np.concatenate([self.direction, np.zeros(1)])
        self.data.mocap_pos[0] = current_pos + delta_pos
    
    def _policy(self, x: np.ndarray) -> np.ndarray:
        self._update_mocap_pos()
        self.agent.set_state(
            time=self.data.time,
            qpos=self.data.qpos,
            qvel=self.data.qvel,
            act=self.data.act,
            mocap_pos=self.data.mocap_pos,
            mocap_quat=self.data.mocap_quat,
            userdata=self.data.userdata,
        )
        planner_step_time = time.time()
        self.agent.planner_step()
        planner_step_time = time.time() - planner_step_time
        print(f"[DEBUG] MujocoMPCService {id(self)} planner step time: {planner_step_time}")
        u = self.agent.get_action(nominal_action=True)
        return u