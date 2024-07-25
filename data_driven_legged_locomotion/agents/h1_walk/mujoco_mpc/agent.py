from mujoco_mpc import agent as agent_lib
import numpy as np
import pathlib

from data_driven_legged_locomotion.common import MujocoService, StateSpace

class MujocoMPCService(MujocoService):
    def __init__(self, ss: StateSpace, model, variances: float = None):
        super().__init__(ss, model, variances)
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
        agent.set_task_parameter("Speed", 0.7)
        self.agent = agent
        self.mocap_pos = np.asarray([2.0, 2.0, 0.25])
    
    def _policy(self, x: np.ndarray) -> np.ndarray:
        self.data.mocap_pos[0] = self.mocap_pos
        self.agent.set_state(
            time=self.data.time,
            qpos=self.data.qpos,
            qvel=self.data.qvel,
            act=self.data.act,
            mocap_pos=self.mocap_pos,
            mocap_quat=self.data.mocap_quat,
            userdata=self.data.userdata,
        )
        self.agent.planner_step()
        u = self.agent.get_action(nominal_action=True)
        return u