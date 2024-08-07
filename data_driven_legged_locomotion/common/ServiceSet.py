from abc import ABC, abstractmethod
import mujoco
import numpy as np

from .StatePF import StateCondPF, NormalStatePF, FakeStateCondPF
from .StateSpace import StateSpace


class Behavior(ABC): #{pi(x_k|x_k-1)}_0:N
    """A Behavior is a sequence of InstantBehaviors that describes the behavior of a system in a finite span of time."""
    def __init__(self, ss: StateSpace, time_window: int):
        self.ss = ss
        self.N = time_window
        self.time_window: list[StateCondPF] = []
    
    def getAtTime(self, k: int) -> StateCondPF:
        """Returns the state conditional probability distribution at time k."""
        if k < 0 or k > self.N - 1:
            raise ValueError(f"Time index k must be between 0 and {self.N-1}.")
        return self.time_window[k]

class Service(ABC):
    """A Service provides a behavior to the crowdsourcing algorithm."""
    def __init__(self, ss: StateSpace):
        self.ss = ss
    
    @abstractmethod
    def _generateBehavior(self, initial_state_index: tuple, N: int) -> Behavior:
        """Generates a behavior for the given state."""
        pass
    
    def generateBehavior(self, state: np.array, N: int) -> Behavior:
        """Generates a behavior for the given state."""
        return self._generateBehavior(self.ss.toIndex(state), N)

class BehaviorSet: #{{pi(x_k|x_k-1)}_0:N}_1:S
    def __init__(self, ss: StateSpace, N: int):
        self.ss = ss
        self.behaviors: list[Behavior] = []
        self.N = N
    
    def __len__(self):
        return len(self.behaviors)
    
    @property
    def S(self):
        return len(self.behaviors)
    
    def add(self, behavior: Behavior):
        """Adds a behavior to the set."""
        if behavior.ss != self.ss:
            raise ValueError("Behavior state space does not match the state space of the BehaviorSet.")
        if behavior.N != self.N:
            raise ValueError("Behavior time window does not match the time window of the BehaviorSet.")
        self.behaviors.append(behavior)
        
    def getAtTime(self, k: int) -> list[StateCondPF]:
        """Returns the state conditional probability distributions at time k."""
        return [behavior.getAtTime(k) for behavior in self.behaviors]
    
    def extractBehavior(self, s_list: list[int]) -> Behavior:
        """Extracts a behavior from a list of service indices."""
        if len(s_list) != self.N:
            raise ValueError("The list of indices must have the same length as the time window.")
        behavior = Behavior(self.ss, self.N)
        for k, s in enumerate(s_list):
            behavior.time_window.append(self.behaviors[s].time_window[k])
        return behavior

class ServiceSet:
    """Abstract class for a set of services."""
    def __init__(self, ss: StateSpace):
        self.ss = ss
        self.services: list[Service] = []
        
    def __len__(self):
        return len(self.services)
    
    def addService(self, service: Service):
        """Adds a service to the set."""
        if service.ss != self.ss:
            raise ValueError("Service state space does not match the state space of the ServiceSet.")
        self.services.append(service)
    
    def getNumServices(self) -> int:
        """Returns the number of services."""
        return len(self.services)
    
    def getBehaviors(self, x_0: np.array, N: int) -> BehaviorSet:
        behavior_set = BehaviorSet(self.ss, N)
        for service in self.services:
            behavior_set.add(service.generateBehavior(x_0, N))
        return behavior_set
    
class SingleBehavior(Behavior):
    def __init__(self, ss: StateSpace, behavior: StateCondPF):
        super().__init__(ss, 1)
        self.time_window.append(behavior)
        
class MujocoService(Service):
    """A service that exploits a deterministic policy in a Mujoco environment to generate behaviors."""
    def __init__(self, ss: StateSpace, model, variances: float = None):
        super().__init__(ss)
        if variances is None:
            variances = np.ones(ss.n_states) * 0.01
        self.variances = variances
        self.model = model
        self.data = mujoco.MjData(model)
        if np.any(variances <= 0):
            raise ValueError("Variance must be positive.")
        model_states = model.nq + model.nv
        if model_states != self.ss.n_states:
            raise ValueError(f"State space dimensions {self.ss.n_states} do not match the Mujoco model {model_states}.")
    
    def policy(self, x: np.array) -> np.array:
        """Returns the control action for the given state."""
        return self._policy(x)
    
    @abstractmethod
    def _policy(self, x: np.array) -> np.array:
        """Returns the control action for the given state."""
        pass
    
    def _generateBehavior(self, initial_state_index: tuple, N: int) -> Behavior:
        """Generates a behavior for the given state."""
        if N > 1:
            raise ValueError("MujocoService only supports N=1.")
        x = self.ss.toState(initial_state_index)
        self.data.qpos = x[0:self.model.nq]
        self.data.qvel = x[self.model.nq:]
        u = self._policy(x)
        self.data.ctrl = u
        mujoco.mj_step(self.model, self.data)
        x_next = np.concatenate([self.data.qpos, self.data.qvel])
        pf = NormalStatePF(self.ss, x_next, np.diag(self.variances))
        cond_pf = FakeStateCondPF(self.ss, pf)
        return SingleBehavior(self.ss, cond_pf)