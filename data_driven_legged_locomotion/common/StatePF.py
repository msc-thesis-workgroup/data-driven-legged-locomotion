from abc import ABC, abstractmethod
import numpy as np

from .StateSpace import StateSpace

class StatePF(ABC): #pi(x_k)
    """A StatePF is a probability distribution over the state space."""
    def __init__(self, ss: StateSpace):
        self.ss = ss
        
    @abstractmethod
    def getProb(self, state_index: tuple) -> float:
        pass
    
    @abstractmethod
    def sample(self) -> np.array:
        pass
    
    @abstractmethod
    def getMean(self) -> np.array:
        pass
    
    @abstractmethod
    def getVariance(self) -> np.array:
        pass
    
    @abstractmethod
    def getEntropy(self) -> float:
        pass
    
    def monteCarloExpectation(self, func: callable, num_samples: int = 50) -> float:
        """Computes the expectation of a function using Monte Carlo sampling."""
        samples = np.array([self.sample() for _ in range(num_samples)])
        return np.mean(func(samples))
    
    def getProbState(self, state: np.array) -> float:
        """Returns the probability of a state."""
        return self.getProb(self.ss.toIndex(state))
        
class StateCondPF(ABC): #pi(x_k|x_k-1)
    """An StateCondPF is a conditional PF that describes the behavior of a system at a given time."""
    def __init__(self, ss: StateSpace):
        self.ss = ss
        self.PFs = {}
    
    @abstractmethod
    def _getNextStatePF(self, state_index: np.array) -> StatePF:
        pass
        
    def getNextStatePF(self, state_index: np.array) -> StatePF:
        """Returns the next state probability distribution given the current state."""
        if not state_index in self.PFs:
            self.PFs[state_index] = self._getNextStatePF(state_index)
        return self.PFs[state_index]

class HistogramStatePF(StatePF):
    """A StatePF that represents a probability distribution over the state space using a histogram."""
    def __init__(self, ss: StateSpace, data: np.array):
        super().__init__(ss)
        if np.any(data.shape != ss.dims):
            raise ValueError("Histogram must have the same shape as the state space.")
        self.histogram = data
        # Normalize the histogram
        self.histogram = self.histogram / np.sum(self.histogram)
        
    def getProb(self, state_index: tuple) -> float:
        return self.histogram[state_index]
    
    def sample(self) -> np.array:
        index_flat = np.random.choice(np.prod(self.ss.dims), p=self.histogram.flatten())
        index = np.unravel_index(index_flat, self.ss.dims)
        index = np.array(index)
        return self.ss.toState(index)
    
    def getMean(self) -> np.array:
        mean = 0
        for index, state in self.ss:
            mean = mean + state * self.getProb(index)
        return mean
    
    def getVariance(self) -> np.array:
        variance = 0
        for index, state in self.ss:
            variance = variance + (state - self.getMean())**2 * self.getProb(index)
        return variance
    
    def getEntropy(self) -> float:
        entropy = 0
        for index, state in self.ss:
            prob = self.getProb(index)
            if prob > 0:
                entropy = entropy - prob * np.log(prob)
        return entropy
    
class NormalStatePF(StatePF):
    """A StatePF that represents a probability distribution over the state space using a normal distribution."""
    def __init__(self, ss: StateSpace, mean: np.array, cov: np.array):
        super().__init__(ss)
        self.mean = mean
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)
        self.det_cov = np.linalg.det(cov)
        
    def getProb(self, state_index: tuple) -> float:
        state = self.ss.toState(state_index)
        return np.exp(-0.5 * (state - self.mean).T @ self.inv_cov @ (state - self.mean)) / np.sqrt((2 * np.pi)**self.ss.n_states * self.det_cov)
    
    def sample(self) -> np.array:
        return np.random.multivariate_normal(self.mean, self.cov)
    
    def getMean(self) -> np.array:
        return self.mean
    
    def getVariance(self) -> np.array:
        return self.cov
    
    def getEntropy(self) -> float:
        return 0.5 * np.log((2 * np.pi * np.e)**self.ss.n_states * self.det_cov)

if __name__ == "__main__":
    bounds = np.array([[-1.0,1.0],[0.0,1.0]])
    deltas = [0.1,0.5]
    ss = StateSpace(2,bounds,deltas)
    data = np.random.rand(*ss.dims)
    data = data / np.sum(data)
    hist_pf = HistogramStatePF(ss, data)
    print(hist_pf.getProb((0,0)))
    print(hist_pf.sample())
    print(hist_pf.getMean())
    print(hist_pf.getVariance())
    print(hist_pf.getEntropy())
    mean = np.array([0.0,0.5])
    cov = np.array([[0.1,0.0],[0.0,0.1]])
    norm_pf = NormalStatePF(ss, mean, cov)
    print(norm_pf.getProb((0,0)))
    print(norm_pf.sample())
    print(norm_pf.getMean())
    print(norm_pf.getVariance())
    print(norm_pf.getEntropy())
    print(norm_pf.monteCarloExpectation(lambda x: x[:,0]**2+x[:,1]**2, num_samples=10000))