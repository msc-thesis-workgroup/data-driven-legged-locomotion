from mujoco_mpc import agent as agent_lib
import numpy as np
import pathlib
import pickle
import scipy
import time

from data_driven_legged_locomotion.common import MujocoService, StateSpace, OfflineReaderService, Behavior, NormalStateCondPF, SingleBehavior

class MujocoMPCService(MujocoService):
    """A serve that uses a Mujoco MPC agent to generate behaviors."""
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
    """This class is a variant of MujocoMPCService that uses the last state of the best trajectory as the next state for
    better distringuishing the services in the crowdsourcing algorithm."""
    def _get_next_state(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        self.data.qpos = state[0:self.model.nq]
        self.data.qvel = state[self.model.nq:]
        self._last_u = self._policy(state, t)
        best_trajectory = self.agent.best_trajectory()
        state_traj = best_trajectory["states"]
        return state_traj[-1,:self.ss.n_states]

class FNOStateSpace(StateSpace):
    def __init__(self):
        super().__init__(3)
    
    @staticmethod
    def quat_to_forward_vector(quat):
        return scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True).as_matrix()[:2,0]
    
    @staticmethod
    def rotate_quat(quat, angle):
        r = scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True)
        r = r * scipy.spatial.transform.Rotation.from_euler('z', angle)
        return r.as_quat(canonical=True, scalar_first=True)
    
    @staticmethod
    def get_fno_from_delta(qpos, qpos_old, delta_time):
        vx = (qpos[0] - qpos_old[0])/delta_time
        vy = (qpos[1] - qpos_old[1])/delta_time
        fw = FNOStateSpace.quat_to_forward_vector(qpos[3:7])
        theta = np.arctan2(fw[1], fw[0])
        v_fw = np.dot([vx, vy], np.array([np.cos(theta), np.sin(theta)]))
        v_n = np.dot([vx, vy], np.array([np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)]))
        fw_old = FNOStateSpace.quat_to_forward_vector(qpos_old[3:7])
        theta_old = np.arctan2(fw_old[1], fw_old[0])
        omega = (theta - theta_old)/delta_time
        return np.array([v_fw, v_n, omega])

class FNOAR2(NormalStateCondPF):
    def __init__(self, ss: StateSpace, policy_coeffs: np.ndarray, policy_cov: np.ndarray):
        super().__init__(ss)
        self.policy_coeffs = policy_coeffs
        self.policy_cov = policy_cov/10000 # This is needed to reduce the number of samples needed for monte carlo, will be removed in the future
    
    def get_mean_cov(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """State is [[v_fw, v_n, omega], [v_fw_old, v_n_old, omega_old]]"""
        if state.shape != (2,3):
            raise ValueError(f"State shape {state.shape} does not match (2,3)")
        curr_fno = state[0]
        old_fno = state[1]
        regressors = np.array([curr_fno[0], old_fno[0], curr_fno[1], old_fno[1], curr_fno[2], old_fno[2], 1])
        means = np.matmul(self.policy_coeffs, regressors)
        return means, self.policy_cov
        

class OfflineAR2Service(OfflineReaderService):
    """This class is a variant of MujocoMPCService that uses an offline reader to generate behaviors.
    The behavior is given by a NormalStateCondPF that represents an AR(2) model."""
    def __init__(self, ss: StateSpace, file_path: str, policy_name: str):
        self.policy_name = policy_name
        super().__init__(ss, file_path)
        
    def _readBehavior(self, file_path: str) -> Behavior:
        """Reads a behavior from a file."""
        models = pickle.load(open(file_path, "rb"))
        self.model_timestep = models['timestep']
        self.cond_pf = FNOAR2(self.ss, models['models'][self.policy_name]['coeffs'], models['models'][self.policy_name]['cov'])
        return SingleBehavior(self.ss, self.cond_pf)
        
if __name__ == "__main__":
    import pathlib
    import matplotlib.pyplot as plt
    ss = FNOStateSpace()
    current_dir = pathlib.Path(__file__).parent
    models_file = current_dir / "models_ols_fno.pkl"
    service_forward = OfflineAR2Service(ss, models_file, "FORWARD")
    service_left = OfflineAR2Service(ss, models_file, "LEFT")
    service_right = OfflineAR2Service(ss, models_file, "RIGHT")
    state_cond_pf_forward = service_forward.generateBehavior(None, N=0).getAtTime(0) # It is not necessary to pass the initial state as it is not used
    state_pf_forward = state_cond_pf_forward.getNextStatePF(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    state_cond_pf_left = service_left.generateBehavior(None, N=0).getAtTime(0) # It is not necessary to pass the initial state as it is not used
    state_pf_left = state_cond_pf_left.getNextStatePF(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    state_cond_pf_right = service_right.generateBehavior(None, N=0).getAtTime(0) # It is not necessary to pass the initial
    state_pf_right = state_cond_pf_right.getNextStatePF(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    print("Forward:")
    print("Mean: ", state_pf_forward.mean)
    print("Cov: ", state_pf_forward.cov)
    print("Left:")
    print("Mean: ", state_pf_left.mean)
    print("Cov: ", state_pf_left.cov)
    print("Right:")
    print("Mean: ", state_pf_right.mean)
    print("Cov: ", state_pf_right.cov)
    fw_samples = state_pf_forward.sample(1000)
    left_samples = state_pf_left.sample(1000)
    right_samples = state_pf_right.sample(1000)
    plt.scatter(0,0, color='k', marker='x')
    plt.quiver(fw_samples[:,0], fw_samples[:,1], np.cos(fw_samples[:,2]), np.sin(fw_samples[:,2]), color='r', alpha=0.5)
    plt.quiver(left_samples[:,0], left_samples[:,1], np.cos(left_samples[:,2]), np.sin(left_samples[:,2]), color='g', alpha=0.5)
    plt.quiver(right_samples[:,0], right_samples[:,1], np.cos(right_samples[:,2]), np.sin(right_samples[:,2]), color='b', alpha=0.5)
    plt.show()