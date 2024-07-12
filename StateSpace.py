class StateSpace:
    """StateSpace provides a representation for the (discretized) state space over which PFs are defined"""

    def __init__(self, n_states: int, bounds: np.array, max_deltas = list[int]):
        """Initializes the state space with the given number of states and bounds.
        max_deltas is a list of the maximum cell size in each dimension."""
        if(len(bounds) != n_states):
            raise("Dimension vector must have the same size of the ranges vector")
        self.n_states = n_states
        self.bounds = bounds
        self.buildStateSpace(max_deltas)

    @property
    def ranges(self):
        return self.bounds[:,1] - self.bounds[:,0]

    def buildStateSpace(self, max_deltas):
        self.dim = self.ranges / max_deltas # Element-wise division
        self.dim = np.ceil(self.dim).astype(int) + 1 # If the division is not an integer, we increase the resolution to make the space uniform
        self.deltas = self.ranges / self.dim # The actual sampling resolution
        # linspaces = []
        # for i in range(self.n_states):
        #     linspaces.append(np.linspace(self.bounds[i,0],self.bounds[i,1],num=self.dim[i], endpoint=True))
        # self.X = np.stack(np.meshgrid(*linspaces,indexing='xy'))

    def toIndex(self, state: np.array):
        """Converts a state to an index in the state space."""
        if len(state) != self.n_states:
            raise ValueError("State has the wrong dimension.")
        if np.any(state < self.bounds[:,0]) or np.any(state > self.bounds[:,1]):
            raise ValueError("State out of bounds.")
        return np.rint((state - self.bounds[:,0]) / self.deltas) # Round to closest cell
    
    def toState(self, index: np.array):
        """Converts an index to a state in the state space."""
        if len(index) != self.n_states:
            raise ValueError("Index has the wrong dimension.")
        if np.any(index < 0) or np.any(index >= self.dim):
            raise ValueError("Index out of bounds.")
        return self.bounds[:,0] + index * self.deltas