
from ..common.StateSpace import StateSpace
from ..common.ServiceSet import MujocoService
import numpy as np
import os

from tdmpc.tdmpc2 import TDMPC2
import hydra

DEFAULT_CONFIG_PATH = "./config/config.yaml"

class TDMPCService():

    def __init__(self, ss: StateSpace, model, agent_path: str,variances: float = None, config_path: str = None):
        super().__init__(ss, model, variances)

        if config_path == "" or config_path is None:
            print("No config path provided. Using default config path.")
            config_path = DEFAULT_CONFIG_PATH
        
        if agent_path == "" or agent_path is None:
            raise ValueError("No agent path provided.")

        self.t = 0
        self.agent = None
        
        self._setup_agent(config_path, agent_path)
        

    def _policy(self, x: np.array) -> np.array:
        """Returns the action given the state."""
        
        action = self.agent.act(x, t0=self.t == 0, task=None)
        self.t += 1
        return action
        
    def _setup_agent(self, x,config_path: str, agent_path: str):
        # Load agent

        # import cfg from config_path
        cfg = hydra.utils.instantiate(config_path) # cfg contains the configuration for the agent (the neural network parameters and so on)
        print("[DEBUG TDMPCService] cfg: ", cfg)

        self.agent = TDMPC2(cfg)
        
        # check if agent path exists
        if not os.path.exists(agent_path):
            raise FileNotFoundError(f"Agent path {agent_path} does not exist.")

        self.agent.load(agent_path)



if __name__ == "__main__":

    TDMPCService(None)