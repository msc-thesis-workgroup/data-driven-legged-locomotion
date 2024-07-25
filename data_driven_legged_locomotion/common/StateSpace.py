import numpy as np

class StateSpace:
    """StateSpace provides a representation for the (discretized) state space over which PFs are defined"""

    def __init__(self, n_states: int, bounds: np.ndarray, max_deltas = list[int]):
        """Initializes the state space with the given number of states and bounds.
        max_deltas is a list of the maximum cell size in each dimension."""
        if(len(bounds) != n_states):
            raise("Dimension vector must have the same size of the ranges vector")
        self.n_states = n_states
        self.bounds = bounds
        self._buildStateSpace(max_deltas)

    def __iter__(self):
        self._iter_state = 0
        return self

    def __next__(self):
        max_index = np.prod(self.dims) - 1
        if self._iter_state > max_index:
            raise StopIteration
        index = np.unravel_index(self._iter_state, self.dims)
        self._iter_state += 1
        return (index, self.toState(index))

    @property
    def ranges(self):
        return self.bounds[:,1] - self.bounds[:,0]
    
    @property
    def total_combinations(self):
        return np.prod(self.dims)

    def _buildStateSpace(self, max_deltas):
        self.dims = self.ranges / max_deltas # Element-wise division
        self.dims = np.ceil(self.dims).astype(int) + 1 # If the division is not an integer, we increase the resolution to make the space uniform
        self.deltas = self.ranges / (self.dims-1) # The actual sampling resolution
        # linspaces = []
        # for i in range(self.n_states):
        #     linspaces.append(np.linspace(self.bounds[i,0],self.bounds[i,1],num=self.dims[i], endpoint=True))
        # self.X = np.stack(np.meshgrid(*linspaces,indexing='xy'))

    def toIndex(self, state: np.ndarray) -> tuple:
        """Converts a state to an index in the state space."""
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
        if state.shape[1] != self.n_states:
            raise ValueError(f"Axis 1 has size {state.shape[1]} but {self.n_states} is expected.")
        if np.any(state < self.bounds[:,0]) or np.any(state > self.bounds[:,1]):
            raise ValueError(f"State {state} out of bounds.")
        res = np.empty(state.shape, dtype=int)
        np.rint((state - self.bounds[:,0]) / self.deltas, casting='unsafe', out=res) # Round to closest cell
        res = res.T # Transpose to have the n_states x n_samples shape
        res = np.squeeze(res) # Remove the extra dimension if the input was a single state
        res = tuple(res) # Convert to tuple
        return res
    
    def toState(self, index: tuple):
        """Converts an index to a state in the state space."""
        index_arr = np.array(index)
        if len(index) != self.n_states:
            raise ValueError("Index has the wrong dimension.")
        if np.any(index_arr < 0) or np.any(index_arr >= self.dims):
            raise ValueError("Index out of bounds.")
        return self.bounds[:,0] + index_arr * self.deltas
    
if __name__ == "__main__":
    bounds = np.array([[-1.0,1.0],[0.0,1.0]])
    deltas = [0.1,0.5]
    ss = StateSpace(2,bounds,deltas)
    for i, state in ss:
        print(i, state)
    i = ss.toIndex(np.array([-1.0,0.25]))
    print(i)
    print(ss.toState(i))
