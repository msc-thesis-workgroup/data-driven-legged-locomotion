import pathlib
import mujoco
import numpy as np
import scipy

from data_driven_legged_locomotion.maps.h1_walk import Map
from data_driven_legged_locomotion.utils.quaternions import quat_to_forward_vector
from data_driven_legged_locomotion.common import StateSpace, MujocoEnvironment, DiscreteStateSpace

class H1WalkEnvironment(MujocoEnvironment):
    def __init__(self, ss = None, custom_model = None):
        if custom_model is None:
            model = self._get_model_path()
        else:
            model = custom_model
        if ss is None:
            ss = StateSpace(26+25)
        super().__init__(ss, model)
        
    def _get_model_path(self):
        base_path = pathlib.Path(__file__).parent.parent.parent.parent
        print("[DEBUG] base_path: ", base_path)
        task_candidates = [path for path in pathlib.Path(base_path).rglob("mujoco_mpc-build/mjpc/tasks/h1/walk/task.xml")]
        print("[DEBUG] task candidates: ", task_candidates)
        if len(task_candidates) == 0:
            raise ValueError(f"Could not find h1_walk task in folder {base_path}, make sure to build mujoco_mpc")
        print("Task found: ", task_candidates[0])
        model_path = task_candidates[0]
        return model_path
    
    @staticmethod
    def get_fno_from_delta(qpos, qpos_old, delta_time):
        vx = (qpos[0] - qpos_old[0])/delta_time
        vy = (qpos[1] - qpos_old[1])/delta_time
        fw = quat_to_forward_vector(qpos[3:7])
        theta = np.arctan2(fw[1], fw[0])
        v_fw = np.dot([vx, vy], np.array([np.cos(theta), np.sin(theta)]))
        v_n = np.dot([vx, vy], np.array([np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)]))
        fw_old = quat_to_forward_vector(qpos_old[3:7])
        theta_old = np.arctan2(fw_old[1], fw_old[0])
        omega = (theta - theta_old)/delta_time
        return np.array([v_fw, v_n, omega])
        
class H1WalkEnvironmentDiscrete(H1WalkEnvironment):
    def __init__(self, n_samples = 1000):
        upper_q_bounds = np.ones(26)*10
        lower_q_bounds = -upper_q_bounds
        upper_dq_bounds = np.ones(6+19)*35
        lower_dq_bounds = -upper_dq_bounds
        upper_bounds = np.concatenate([upper_q_bounds, upper_dq_bounds])
        lower_bounds = np.concatenate([lower_q_bounds, lower_dq_bounds])
        deltas_q = (upper_q_bounds - lower_q_bounds) / n_samples
        deltas_dq = (upper_dq_bounds - lower_dq_bounds) / n_samples
        deltas = np.concatenate([deltas_q, deltas_dq])
        ss = DiscreteStateSpace(26+25,
                        np.array(list(zip(lower_bounds, upper_bounds))),
                        deltas)
        super().__init__(ss)
        
class H1WalkMapEnvironment(H1WalkEnvironment):
    def __init__(self, map: Map):
        self.map = map
        model_spec = mujoco.MjSpec()
        model_spec.from_file(str(self._get_model_path()))
        map.add_to_spec(model_spec)
        model = model_spec.compile()
        self.trigger = False
        super().__init__(custom_model=model)
    
    def update_dynamic_obs(self):
        if not self.trigger:
            return
        self.map.step(self.model, self.timestep)
    
    def trigger(self):
        self.trigger = True
    
    def step(self, u: np.ndarray|float):
        MujocoEnvironment.step(self, u)
        self.update_dynamic_obs()
        
    