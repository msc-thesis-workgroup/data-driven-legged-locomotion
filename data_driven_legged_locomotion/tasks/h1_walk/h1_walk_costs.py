import numpy as np

from data_driven_legged_locomotion.tasks.h1_walk import H1WalkEnvironment
from data_driven_legged_locomotion.utils.paths import closest_point_on_path, path_tangent_vectors
from data_driven_legged_locomotion.utils.quaternions import quat_to_forward_vector, rotate_quat

def h1_quadratic_objective(x, k):
    """x is the full state of the robot, k is the time step."""
    r = np.array([10.0, 10.0])
    costs = (x[:,0] - r[0])**2 + (x[:,1] - r[1])**2
    costs = np.squeeze(costs)
    return costs

class H1TrackCost:
    def __init__(self, env: H1WalkEnvironment, path: np.ndarray = None, timestep: float = 0.02):
        self.env = env
        self.path = path
        self.timestep = timestep
        self.lookahead_steps_xy = 500
        self.lookahead_steps_angle = 50
        
    def update_path(self, path):
        self.path = path
    
    @property
    def path(self):
        return self._path
    
    @path.setter
    def path(self, value):
        self._path = value
        self._path_d = path_tangent_vectors(value)
    
    @staticmethod
    def crowdsourcing_cost(x, y, forward_vector, path, path_d):
        x = np.array([x,y])
        closest_point, crosstrack_error, tangent_vector = closest_point_on_path(x, path, path_d)
        orientation_error = (np.dot(forward_vector, tangent_vector)-1)**2
        return (crosstrack_error + 10*orientation_error)*100
    
    def __call__(self, samples, k):
        """Cost function for the H1 track task. 
            samples is [[forward_velocity, lateral_velocity, angular_velocity]].
            k is the time step."""
        if self._path is None:
            raise ValueError("Path is not set.")
        fw = quat_to_forward_vector(self.env.data.qpos[3:7])
        theta = np.arctan2(fw[1], fw[0])
        future_pos_x = self.env.data.qpos[0] + samples[:,0]*np.cos(theta)*self.timestep*self.lookahead_steps_xy + samples[:,1]*np.sin(theta)*self.timestep*self.lookahead_steps_xy
        future_pos_y = self.env.data.qpos[1] + samples[:,0]*np.sin(theta)*self.timestep*self.lookahead_steps_xy - samples[:,1]*np.cos(theta)*self.timestep*self.lookahead_steps_xy
        fw_vectors = [quat_to_forward_vector(rotate_quat(self.env.data.qpos[3:7], angle)) for angle in samples[:,2]*self.timestep*self.lookahead_steps_angle]
        cost_samples = [H1TrackCost.crowdsourcing_cost(x[0], x[1], x[2], self._path, self._path_d) for x in zip(future_pos_x, future_pos_y, fw_vectors)]
        return np.mean(cost_samples)