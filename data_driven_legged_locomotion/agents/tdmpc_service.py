
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
        action = self.agent.act(x, t0=self.t == 0, task=None)
        self.t += 1
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

    def _generalize_walk_direction(self,obs: np.array):
        
        transformation_quat = self.transformation_quat

        current_quat = Quaternion(obs[3:7])  # Convert tensor slice to numpy array for Quaternion
        current_position = obs[0:3] # Convert tensor slice to numpy array for Quaternion

        new_quat = transformation_quat * current_quat
        new_pos = transformation_quat.rotate(current_position)
        new_vel = transformation_quat.rotate(obs[26:29])

        obs[0:3] = torch.from_numpy(new_pos).type_as(torch.FloatTensor)
        obs[3:7] = torch.from_numpy(new_quat.q).type_as(torch.FloatTensor)
        obs[26:29] = torch.from_numpy(new_vel).type_as(torch.FloatTensor)

        return obs
