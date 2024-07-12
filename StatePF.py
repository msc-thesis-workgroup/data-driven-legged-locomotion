import numpy as np

from StateSpace import StateSpace

class StatePF: #pi(x_k)
    """A StatePF is a probability distribution over the state space."""
    def __init__(self, ss: StateSpace):
        self.ss = ss
        self.p = np.zeros(ss.dim)