from abc import ABC, abstractmethod
import numpy as np

from .StatePF import StateCondPF
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