from abc import ABC, abstractmethod
import numpy as np

from StatePF import StateCondPF
from StateSpace import StateSpace


class Behavior(ABC): #{pi(x_k|x_k-1)}_0:N
    """A Behavior is a sequence of InstantBehaviors that describes the behavior of a system in a finite span of time."""
    def __init__(self, ss: StateSpace, time_window: int):
        self.ss = ss
        self.N = time_window
        self.time_window: list[StateCondPF] = []
    
    def getAtTime(self, k: int) -> StateCondPF:
        """Returns the state conditional probability distribution at time k."""
        return self.time_window[k]

class Service(ABC):
    """A Service provides a behavior to the crowdsourcing algorithm."""
    def __init__(self, ss: StateSpace):
        self.ss = ss
    
    @abstractmethod
    def _generateBehavior(self, initial_state_index: tuple) -> Behavior:
        """Generates a behavior for the given state."""
        pass
    
    def generateBehavior(self, state: np.array) -> Behavior:
        """Generates a behavior for the given state."""
        return self._generateBehavior(self.ss.toIndex(state))

class BehaviorSet: #{{pi(x_k|x_k-1)}_0:N}_1:S
    def __init__(self, ss: StateSpace, N: int):
        self.ss = ss
        self.behaviors: list[Behavior] = []
        self.N = N
    
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
        
    def f(self):
        for k in range(self.N, 0, -1):
            for s in self.S:
                self.behaviors[s].getAtTime(k)

class ServiceSet:
    """Abstract class for a set of services."""
    def __init__(self, ss: StateSpace):
        self.ss = ss
        self.services: list[Service] = []
    
    def addService(self, service: Service):
        """Adds a service to the set."""
        if service.ss != self.ss:
            raise ValueError("Service state space does not match the state space of the ServiceSet.")
        self.services.append(service)
    
    def getNumServices(self) -> int:
        """Returns the number of services."""
        return len(self.services)
    
    def getBehaviors(self, x_0: np.array) -> BehaviorSet:
        behavior_set = BehaviorSet()
        for service in self.services:
            behavior_set.add(service.generateBehavior(x_0))
        return behavior_set
    