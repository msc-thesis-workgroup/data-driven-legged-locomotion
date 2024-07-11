from abc import ABC, abstractmethod
import numpy as np

class StatePF(ABC): #pi(x_k)
    """A StatePF is a probability distribution over the state space."""
    pass

class StateCondPF(ABC): #pi(x_k|x_k-1)
    """An StateCondPF is a conditional PF that describes the behavior of a system at a given time."""
    pass

class Behavior(ABC): #{pi(x_k|x_k-1)}_0:N
    """A Behavior is a sequence of InstantBehaviors that describes the behavior of a system in a finite span of time."""
    def __init__(self, time_window: int):
        self.N = time_window
        self.time_window: list[StateCondPF] = []
    
    def get_state_cond_pf(self, k: int) -> StateCondPF:
        """Returns the state conditional probability distribution at time k."""
        return self.time_window[k]

class BehaviorSet(ABC): #{{pi(x_k|x_k-1)}_0:N}_1:S
    def __init__(self, N: int):
        self.behaviors: list[Behavior] = []
        self.N = N
    
    @property
    def S(self):
        return len(self.behaviors)
    
    def add_behavior(self, behavior: Behavior):
        """Adds a behavior to the set."""
        if behavior.N != self.N:
            raise ValueError("Behavior time window does not match the time window of the BehaviorSet.")
        self.behaviors.add(behavior)
        
    def f(self):
        for t in range(self.N, 0, -1):
            for s in self.S:
                self.behaviors[s].get_state_cond_pf(t)
            

class Service(ABC):
    """A Service provides a behavior to the crowdsourcing algorithm."""
    def generate_behavior(self, x_0: np.array) -> Behavior:
        """Generates a behavior for the given state."""
        pass

class ServiceSet(ABC):
    """Abstract class for a set of services."""
    def __init__(self):
        self.services = set()
    def add_service(self, service: Service):
        """Adds a service to the set."""
        pass
    def get_num_services(self) -> int:
        """Returns the number of services."""
        pass
    def generate_behavior_set(self, x_0: np.array) -> BehaviorSet:
        behavior_set = BehaviorSet()
        for service in self.services:
            behavior_set.add(service.generate_behavior(x_0))
    
# class ContinuousPolicy(ABC):
#     pass

# class DiscretePolicy(ABC):
#     pass