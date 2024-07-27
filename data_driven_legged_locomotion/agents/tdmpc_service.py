
from omegaconf import OmegaConf
from data_driven_legged_locomotion.common.StateSpace import StateSpace
from data_driven_legged_locomotion.common.ServiceSet import MujocoService
import numpy as np
import os

from data_driven_legged_locomotion.agents.tdmpc.tdmpc2 import TDMPC2

import torch
from pyquaternion import Quaternion

DEFAULT_CONFIG_PATH = "./config/config.yaml"
 
DEFAULT_TARGET_DIRECTION = np.array([0.0, 0.0, 0.98, 1.0, 0.0, 0.0, 0.0]) # The neural network is trained with this reference to understand the direction of the movement.
DEFAULT_REFERENCE_DIRECTION = DEFAULT_TARGET_DIRECTION # The target direction of the movement.

class TDMPCService(MujocoService):
    

    def __init__(self, ss: StateSpace, model, agent_path: str,variances: float = None, config_path: str = None):
        super().__init__(ss, model, variances)

        if config_path == "" or config_path is None:
            print("No config path provided. Using default config path.")
            config_path = DEFAULT_CONFIG_PATH
        
        if agent_path == "" or agent_path is None:
            raise ValueError("No agent path provided.")

        self.t = 0
        self.agent = None
        self.target_reference = DEFAULT_TARGET_DIRECTION
        self.transformation_quat = None
        self.policy_reference = None

        self.set_policy_reference(DEFAULT_REFERENCE_DIRECTION)
        self._setup_agent(config_path, agent_path)


    def get_joint_torques(self, action: np.array) -> np.array:
        data = self.data
        model = self.model

        kp = np.array([200, 200, 200, 300, 40, 200, 200, 200, 300, 40, 300, 100, 100, 100, 100, 100, 100, 100, 100])
        kd = np.array([5, 5, 5, 6, 2, 5, 5, 5, 6, 2, 6, 2, 2, 2, 2, 2, 2, 2, 2])
        action_high = np.array([0.43, 0.43, 2.53, 2.05, 0.52, 0.43, 0.43, 2.53, 2.05, 0.52, 2.35, 2.87, 3.11, 4.45, 2.61, 2.87, 0.34, 1.3, 2.61])
        action_low = np.array([-0.43, -0.43, -3.14, -0.26, -0.87, -0.43, -0.43, -3.14, -0.26, -0.87, -2.35, -2.87, -0.34, -1.3,  -1.25, -2.87, -3.11, -4.45, -1.25])

        ctrl = (action + 1) / 2 * (action_high - action_low) + action_low
                
        actuator_length = data.actuator_length
        error = ctrl - actuator_length
        m = model
        d = data

        empty_array = np.zeros(m.actuator_dyntype.shape)
        
        ctrl_dot = np.zeros(m.actuator_dyntype.shape) if np.array_equal(m.actuator_dyntype,empty_array) else d.act_dot[m.actuator_actadr + m.actuator_actnum - 1]
        
        error_dot = ctrl_dot - data.actuator_velocity
        
        joint_torques = kp*error + kd*error_dot

        return joint_torques

    def set_policy_reference(self, policy_reference: np.array):
        """Sets the policy reference. The policy reference is the desired direction of the movement for the agent."""
        self.policy_reference = policy_reference

        # Calculate the transformation matrix from the home orientation to the target orientation
        if len(policy_reference) == 7:
            target_quat = Quaternion(policy_reference[3:7])
        elif len(policy_reference) == 4:
            target_quat = Quaternion(policy_reference)
        else:
            raise ValueError("The policy reference must be a quaternion.")
        
        if len(self.target_reference) == 7:
            target_quat = Quaternion(self.target_reference[3:7])
        elif len(self.target_reference) == 4:
            target_quat = Quaternion(self.target_reference)
        else:
            raise ValueError("The target reference must be a quaternion.")

        self.transformation_quat = target_quat * target_quat.inverse

    def _policy(self, x: np.array) -> np.array:
        """Returns the action given the state."""
        
        x = self._generalize_walk_direction(x)
        x = torch.tensor(x, dtype=torch.float32)
        #print("X: ", x)
        action = self.agent.act(x, t0=self.t == 0, task=None)
        self.t += 1
        action = action.detach().numpy()
        #print("Action pre: ", action)
        action = self.get_joint_torques(action)
        #print("Action after: ", action)
        return action
    
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
