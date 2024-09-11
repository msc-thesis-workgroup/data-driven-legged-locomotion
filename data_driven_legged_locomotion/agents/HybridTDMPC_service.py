
from omegaconf import OmegaConf
from data_driven_legged_locomotion.common.StateSpace import StateSpace
from data_driven_legged_locomotion.common.ServiceSet import MujocoService, Behavior, SingleBehavior, NormalStatePF, FakeStateCondPF
import numpy as np
import os
import pathlib
from data_driven_legged_locomotion.agents.tdmpc.tdmpc2 import TDMPC2

import torch
from pyquaternion import Quaternion
import copy
import mujoco

import gc


DEFAULT_CONFIG_PATH = "./config_h1/config.yaml"
 
DEFAULT_TARGET_DIRECTION = np.array([0.0, 0.0, 0.98, 1.0, 0.0, 0.0, 0.0]) # The neural network is trained with this reference to understand the direction of the movement.
DEFAULT_REFERENCE_DIRECTION = DEFAULT_TARGET_DIRECTION # The target direction of the movement.


def unnorm_action(action: np.array) -> np.array:
    """Unnormalize the action."""
    action_high = np.array([0.43, 0.43, 2.53, 2.05, 0.52, 0.43, 0.43, 2.53, 2.05, 0.52, 2.35, 2.87, 3.11, 4.45, 2.61, 2.87, 0.34, 1.3, 2.61])
    action_low = np.array([-0.43, -0.43, -3.14, -0.26, -0.87, -0.43, -0.43, -3.14, -0.26, -0.87, -2.35, -2.87, -0.34, -1.3,  -1.25, -2.87, -3.11, -4.45, -1.25])
    return (action + 1) / 2 * (action_high - action_low) + action_low


def _compute_joint_torques(data: mujoco.MjData, model: mujoco.MjModel, desired_q_pos: np.array) -> np.array:
    d = data
    m = model

    #self.kp = np.array([200, 200, 200, 300, 40, 200, 200, 200, 300, 40, 300, 100, 100, 100, 100, 100, 100, 100, 100])
    kp = np.array([50, 50, 50, 75, 10, 50, 50, 50, 75, 10, 75, 100, 100, 100, 100, 100, 100, 100, 100])
    
    #self.kd = np.array([5, 5, 5, 6, 2, 5, 5, 5, 6, 2, 6, 2, 2, 2, 2, 2, 2, 2, 2])
    kd = np.array([1.25, 1.25, 1.25, 1.5, 0.25, 1.25, 1.25, 1.25, 1.5, 0.25, 1, 2, 2, 2, 2, 2, 2, 2, 2])
    
    actuator_length = data.qpos[7:len(data.qpos)]
    assert len(actuator_length) == len(desired_q_pos)
    error = desired_q_pos - actuator_length
    m = model
    d = data

    empty_array = np.zeros(m.actuator_dyntype.shape)

    ctrl_dot = np.zeros(m.actuator_dyntype.shape) if np.array_equal(m.actuator_dyntype,empty_array) else d.act_dot[m.actuator_actadr + m.actuator_actnum - 1]

    error_dot = ctrl_dot - data.qvel[6:len(data.qvel)]
    assert len(error_dot) == len(error)

    joint_torques = kp*error + kd*error_dot

    return joint_torques

class HybridTDMPCService(MujocoService):

    def __init__(self, ss: StateSpace, model,variances: float = None, agent_horizon: int = 1, frame_skip: int = 5):
        super().__init__(ss, model, variances)
        
        base_path = pathlib.Path(__file__).parent
        print("[DEBUG] base_path: ", base_path)
        config_path_candidates = [path for path in pathlib.Path(base_path).rglob("hybrid_config.yaml")]
        print("[DEBUG] server_binary_candidates: ", config_path_candidates)
        
        agent_path_candidates = [path for path in pathlib.Path(base_path).rglob("hybrid.pt")]
        print("[DEBUG] agent_path_candidates: ", agent_path_candidates)
        

        if len(config_path_candidates) == 0:
            raise ValueError(f"Could not find agent_server binary in folder {base_path}, make sure to build the agent_server")
        
        if len (config_path_candidates) > 1:
            raise ValueError(f"Multiple config files found in folder {base_path}.")

        if len(agent_path_candidates) == 0:
            raise ValueError(f"Could not find agent file in folder {base_path}, make sure to build the agent_server")
        
        if len (agent_path_candidates) > 1:
            raise ValueError(f"Multiple agent files found in folder {base_path}.")

        config_path = config_path_candidates[0]
        agent_path = agent_path_candidates[0]

        # if config_path == "" or config_path is None:
        #     print("No config path provided. Using default config path.")
        #     config_path = DEFAULT_CONFIG_PATH
        
        # if agent_path == "" or agent_path is None:
        #     raise ValueError("No agent path provided.")

        self.t = 0
        self.agent = None
        self.target_reference = DEFAULT_TARGET_DIRECTION
        self.transformation_quat = None
        self.policy_reference = None
        self.agent_horizon = agent_horizon
        self.control_trajectory = []
        self.frame_skip = frame_skip

        self.set_policy_reference(DEFAULT_REFERENCE_DIRECTION)
        self._setup_agent(config_path, agent_path)


    def _get_joint_torques(self, action: np.array) -> np.array:
        return _compute_joint_torques(data=self.data, model=self.model, desired_q_pos=action)

    def set_data(self, data):
        self.data = data

    def set_policy_reference(self, policy_reference: np.array):
        """Sets the policy reference. The policy reference is the desired direction of the movement for the agent."""
        self.policy_reference = policy_reference

        # Calculate the transformation matrix from the home orientation to the target orientation
        if len(policy_reference) == 7:
            policy_reference = Quaternion(policy_reference[3:7])
        elif len(policy_reference) == 4:
            policy_reference = Quaternion(policy_reference)
        else:
            raise ValueError("The policy reference must be a quaternion.")
        
        if len(self.target_reference) == 7:
            target_quat = Quaternion(self.target_reference[3:7])
        elif len(self.target_reference) == 4:
            target_quat = Quaternion(self.target_reference)
        else:
            raise ValueError("The target reference must be a quaternion.")

        self.transformation_quat = target_quat * policy_reference.inverse

    def _policy(self, x: np.array, t: float = 0.0) -> np.array:
        """Returns the action given the state."""
        
        raise NotImplementedError("The policy method must be implemented.")
        # TD-MPC is trained to walk in the direction of the target reference, however, if the position of the robot is far from the target reference, the agent will produce high torques to move the robot to the target reference. This is not desired because the robot will become unstable. To avoid this, we set the position of the robot to (0,0) so the agent can produce the correct torques to move the robot in the desired direction.
        x = copy.deepcopy(x)
        x[0] = 0.0
        x[1] = 0.0

        # Hybrid TD-MPC: The agent sees only a subset of x.
        x = self._generalize_walk_direction(x).numpy()
        
        x_tdmpc = self._convert_stato_to_TDMPC_state(x)
        x_tdmpc = torch.tensor(x_tdmpc, dtype=torch.float32)
        action = self.agent.act(x_tdmpc, t0=self.t == 0, task=None)
        self.t += 1
        action = action.detach().numpy()
        action = np.concatenate([action, np.zeros(8)])
        desired_joint_pos = unnorm_action(action)
        desired_joint_pos[-8:] = np.zeros(8)
        
        u = self._get_joint_torques(desired_joint_pos)

        return u
    

    # Override
    def _get_next_state(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Returns the next state given the current state and the current time. This method must set _last_u
        as the control action used to reach the next state."""
        
        x = copy.deepcopy(state)
        #data_copy = copy.deepcopy(self.data)
        data_copy = self.data
        agent_copy = self.agent

        x[0] = 0.0
        x[1] = 0.0

        self.control_trajectory = []
        
        for _ in range(self.agent_horizon):
            # Hybrid TD-MPC: The agent sees only a subset of x.
            x = self._generalize_walk_direction(x).numpy()
            
            x_tdmpc = self._convert_stato_to_TDMPC_state(x)
            x_tdmpc = torch.tensor(x_tdmpc, dtype=torch.float32)
            action = agent_copy.act(x_tdmpc, t0=self.t == 0, task=None)
            self.t += 1
            action = action.detach().numpy()
            action = np.concatenate([action, np.zeros(8)])
            desired_joint_pos = unnorm_action(action)
            desired_joint_pos[-8:] = np.zeros(8)
            
            for _ in range(self.frame_skip):
                u = _compute_joint_torques(data=data_copy, model=self.model, desired_q_pos=desired_joint_pos)
                self.control_trajectory.append(u.copy())
                data_copy.ctrl = u.copy()
                mujoco.mj_step(self.model, data_copy)

            x = np.concatenate([data_copy.qpos, data_copy.qvel])
            x[0] = 0.0
            x[1] = 0.0

        next_state = np.concatenate([data_copy.qpos, data_copy.qvel])
        # self._last_u = u
        self.data.qpos = copy.deepcopy(state[0:self.model.nq])
        self.data.qvel = copy.deepcopy(state[self.model.nq:])
        self.data.time = t
        self.data.ctrl = u.copy()
        return next_state

    def _convert_stato_to_TDMPC_state(self, x: np.array) -> np.array:
        x_tdmpc = np.concatenate([x[0:18], x[26:43]]) # x_tdmpc = [x[0:18], x[26:43]] [x[0:26-8], x[26:51-8]]
        #print("x_tdmpc: ", x_tdmpc, "len(x_tdmpc): ", len(x_tdmpc))
        return x_tdmpc
    
    def _setup_agent(self,config_path: str, agent_path: str):
        
        # check if config path exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config path {config_path} does not exist.")

        # check if agent path exists
        if not os.path.exists(agent_path):
            raise FileNotFoundError(f"Agent path {agent_path} does not exist.")

        # Load the configuration file
        cfg = OmegaConf.load(config_path)

        # Create the TD-MPC agent
        self.agent = TDMPC2(cfg)        
        
        # Load agent
        self.agent.load(agent_path)
        print("[DEBUG] Agent loaded.")
        
    def _generalize_walk_direction(self,obs: np.array):
        
        # TODO Convert types to Float 64 to avoid possible quantization errors

        transformation_quat = self.transformation_quat

        current_quat = Quaternion(obs[3:7])  # Convert tensor slice to numpy array for Quaternion
        current_position = obs[0:3] # Convert tensor slice to numpy array for Quaternion

        new_quat = (transformation_quat * current_quat).elements.astype(float)
        new_pos = transformation_quat.rotate(current_position).astype(float)
        new_vel = transformation_quat.rotate(obs[26:29]).astype(float)
        
        # convert obs to tensor
        obs = torch.from_numpy(obs).type(torch.FloatTensor)
        #obs = torch.tensor(obs, dtype=torch.float32)

        obs[0:3] = torch.from_numpy(new_pos).type(torch.FloatTensor)
        obs[3:7] = torch.from_numpy(new_quat).type(torch.FloatTensor)
        obs[26:29] = torch.from_numpy(new_vel).type(torch.FloatTensor)

        return obs
    
    def get_agent_copy(self):
        return self.agent.copy()

    def set_agent_copy(self, agent_copy):
        self.agent = agent_copy

# class HybridTDMPCServiceV2(HybridTDMPCService):
#     def __init__(self, ss: StateSpace, model, variances: float = None, agent_horizon: int = 1):
#         super().__init__(ss, model, variances)
#         self.agent_horizon = agent_horizon
    
#     def set_data(self, data):
#         self.data = data

#     def _get_next_state(self, state: np.ndarray, t: float = 0) -> np.ndarray:
        
#         self.agent_copy = self.agent #.copy()
#         data = copy.deepcopy(self.data)

#         # TD-MPC is trained to walk in the direction of the target reference, however, if the position of the robot is far from the target reference, the agent will produce high torques to move the robot to the target reference. This is not desired because the robot will become unstable. To avoid this, we set the position of the robot to (0,0) so the agent can produce the correct torques to move the robot in the desired direction.        
#         state = self._generalize_walk_direction(state)
#         state = self._convert_stato_to_TDMPC_state(state)

#         state[0] = 0.0
#         state[1] = 0.0

#         state = torch.tensor(state, dtype=torch.float32)
#         action = self.agent_copy.act(state, t0=t == 0, task=None)
#         action = action.detach().numpy()
#         action = np.concatenate([action, np.zeros(8)])
#         action = unnorm_action(action)
#         action[-8:] = np.zeros(8)
        
#         self.control_trajectory = []

#         for i in range(self.agent_horizon):
#             #t += 0.002
#             u = _compute_joint_torques(data=data, model=self.model, desired_q_pos=action)
            
#             self.control_trajectory.append(u)

#             data.ctrl = u
#             mujoco.mj_step(self.model, data)

#             self._last_u = u
        

#         state = np.concatenate([data.qpos, data.qvel])
#         del data
#         gc.collect()
#         return state
    
#     def get_agent_copy(self):
#         return self.agent_copy

#     def set_agent_copy(self, agent_copy):
#         self.agent = agent_copy
#         del self.agent_copy
#         gc.collect()
