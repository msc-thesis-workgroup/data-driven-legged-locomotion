from abc import ABC, abstractmethod
import cvxpy as cp
import numpy as np

from ServiceSet import ServiceSet
from StatePF import StatePF
from StateSpace import StateSpace

class CrowdsourcingBase(ABC):
    def __init__(self, ss: StateSpace, services: ServiceSet, cost: callable, N: int = 1):
        self.ss = ss
        self.services = services
        self.cost = cost
        self.N = N
        self.initialized = False
        
    @property
    def S(self):
        return len(self.services)
        
    def initialize(self, initial_state: np.array):
        self._a = np.zeros((self.N + 1, self.S, self.ss.total_combinations))
        self._alpha = np.zeros((self.N + 1, self.S))
        #self._overline_r = np.zeros((self.N, self.ss.total_combinations))
        self._behaviors = self.services.getBehaviors(initial_state, self.N)
        self._initial_state = initial_state
        self.initialized = True
    
    @abstractmethod
    def _get_DKL(self, pi: StatePF) -> float:
        pass
    
    def _solveOptimization(self, a: np.array):
        alpha = cp.Variable(self.S)
        constraints = [alpha >= 0, cp.sum(alpha) == 1]
        objective = cp.Maximize(cp.sum(cp.multiply(a, alpha)))
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return alpha.value
    
    def run(self):
        if not self.initialized:
            raise ValueError("Crowdsourcing must be initialized first.")
        for k in range(self.N-1, 0-1, -1): # From N-1 to 0
            def eval_r_overline(x):
                x_index_flat = np.ravel_multi_index(x, self.ss.dims)
                return - np.dot(self._a[k+1, :, x_index_flat], self._alpha[k+1, :]) - self.cost(x,k)
            
            # When we are at the end of the recursion, we don't need to evaluate the optimal policy from any state but the initial state
            if k == 0:
                x, x_index = self._initial_state, self.ss.toIndex(self._initial_state)
                behaviors = self._behaviors.getAtTime(k)
                for s in range(self.S):
                    pi_cond = behaviors[s]
                    pi = pi_cond.getNextStatePF(x_index)
                    exp_r_overline = pi.monteCarloExpectation(eval_r_overline)
                    self._a[k, s, x_index_flat] = self._get_DKL(pi) + exp_r_overline
                self._alpha[k,:] = self._solveOptimization(self._a[k, :, x_index_flat])
                break
            
            for x_index, x in self.ss:
                x_index_flat = np.ravel_multi_index(x_index, self.ss.dims)
                #r_hat = - np.dot(self._a[k+1, :, x_index_flat], self._alpha[k+1, :])
                #self._overline_r[k, x_index_flat] = r_hat - self.cost(x,k)
                behaviors = self._behaviors.getAtTime(k)
                for s in range(self.S):
                    pi_cond = behaviors[s]
                    pi = pi_cond.getNextStatePF(x_index)
                    exp_r_overline = pi.monteCarloExpectation(eval_r_overline)
                    self._a[k, s, x_index_flat] = self._get_DKL(pi) + exp_r_overline
                self._alpha[k,:] = self._solveOptimization(self._a[k, :, x_index_flat])