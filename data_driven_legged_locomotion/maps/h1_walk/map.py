from abc import ABC, abstractmethod
import functools
import mujoco
import numpy as np
import scipy

class Obstacle(ABC):
    def __init__(self):
        """
        Initializes the Obstacle class.
        """
        pass
    
    @abstractmethod
    def cost(self, pos: np.ndarray) -> float:
        """
        Computes the cost associated with the given position.

        Args:
            pos (np.ndarray): The position for which to compute the cost.

        Returns:
            float: The computed cost.
        """
        raise NotImplementedError("Method not implemented.")
    
    @abstractmethod
    def add_to_spec(self, model_spec: dict):
        """
        Adds the obstacle to the model specification.

        Args:
            model_spec (dict): The model specification to which the obstacle is added.
        """
        raise NotImplementedError("Method not implemented.")
    
class DynamicObstacle(Obstacle):
    def __init__(self):
        """
        Initializes the DynamicObstacle class.
        """
        self.transition_end = False
    
    @abstractmethod
    def step(self, delta_t: float):
        """
        Advances the obstacle's state by a given time step.

        Args:
            delta_t (float): The time step by which to advance the obstacle's state.
        """
        raise NotImplementedError("Method not implemented.")

class Cylinder(Obstacle):
    last_id = 0
    
    def __init__(self, pos: np.ndarray, radius: float = 0.8, height: float = 0.5):
        """
        Initializes the Cylinder class.

        Args:
            pos (np.ndarray): The position of the cylinder.
            radius (float, optional): The radius of the cylinder. Defaults to 0.8.
            height (float, optional): The height of the cylinder. Defaults to 0.5.
        """
        super().__init__()
        self.id = Cylinder.last_id
        Cylinder.last_id += 1
        self.radius = radius
        self.height = height
        self.pos = pos

    def cost(self, pos: np.ndarray) -> float:
        """
        Computes the cost associated with the given position.

        Args:
            pos (np.ndarray): The position for which to compute the cost.

        Returns:
            float: The computed cost.
        """
        return 30 * scipy.stats.multivariate_normal.pdf(pos, mean=self.pos, cov=0.4 * np.eye(2))
    
    def add_to_spec(self, model_spec: dict):
        """
        Adds the cylinder to the model specification.

        Args:
            model_spec (dict): The model specification to which the cylinder is added.
        """
        body = model_spec.worldbody.add_body()
        body.name = f"obstacle_{self.id}"
        body.pos = self.pos.tolist() + [self.height]
        geom = body.add_geom()
        geom.name = f"obstacle_geom_{self.id}"
        geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        geom.size = [self.radius, self.radius, self.height]
        geom.rgba = [1, 0, 0, 1]

class SlidingWall(DynamicObstacle):
    last_id = 0
    
    def __init__(self, start_pos: np.ndarray, end_pos: np.ndarray, shift_vect: np.ndarray, width: float = 0.2, height: float = 1.0):
        """
        Initializes the SlidingWall class.

        Args:
            start_pos (np.ndarray): The starting position of the wall.
            end_pos (np.ndarray): The ending position of the wall.
            shift_vect (np.ndarray): The shift vector for the wall.
            width (float, optional): The width of the wall. Defaults to 0.2.
            height (float, optional): The height of the wall. Defaults to 1.0.
        """
        super().__init__()
        self.id = SlidingWall.last_id
        SlidingWall.last_id += 1
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.shift_vect = shift_vect
        self.width = width
        self.height = height
        self.length = np.linalg.norm(end_pos - start_pos)
        self.yaw = np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        self.mean_point = (start_pos + end_pos) / 2
        self.pos = np.zeros(2)  # Position of the center of the obstacle
        
    def cost(self, pos: np.ndarray) -> float:
        """
        Computes the cost associated with the given position.

        Args:
            pos (np.ndarray): The position for which to compute the cost.

        Returns:
            float: The computed cost.
        """
        n_obs = max(1, int(self.length / 1.0))
        obs_perc = np.linspace(0, 1, n_obs)
        start_point, end_point = self.start_pos, self.end_pos
        if self.transition_end:
            start_point = start_point + self.shift_vect
            end_point = end_point + self.shift_vect
        c = 0.0
        for perc in obs_perc:
            obs_point = start_point + perc * (end_point - start_point)
            c += 100 * scipy.stats.multivariate_normal.pdf(pos, mean=obs_point, cov=0.2 * np.eye(2))
        return c
    
    def add_to_spec(self, model_spec: dict):
        """
        Adds the sliding wall to the model specification.

        Args:
            model_spec (dict): The model specification to which the sliding wall is added.
        """
        body = model_spec.worldbody.add_body()
        body.name = f"obstacle_swall_{self.id}"
        body.pos = self.mean_point.tolist() + [self.height]
        body.quat = scipy.spatial.transform.Rotation.from_euler('z', self.yaw).as_quat(scalar_first=True)
        geom = body.add_geom()
        geom.name = f"obstacle_swall_geom_{self.id}"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [self.length / 2, self.width, self.height]
        geom.rgba = [1, 0, 0, 1]
        
    def step(self, model: mujoco.MjModel, delta_t: float):
        """
        Advances the wall's state by a given time step.

        Args:
            model (mujoco.MjModel): The MuJoCo model.
            delta_t (float): The time step by which to advance the wall's state.
        """
        if self.transition_end:
            return
        translation = np.linalg.norm(self.shift_vect)
        direction = self.shift_vect / translation
        if np.dot(direction, self.pos) >= translation:
            self.transition_end = True
            return
        self.pos += direction * delta_t * 10.0
        start_point, end_point = self.start_pos, self.end_pos
        mean_point = (start_point + end_point) / 2
        new_mean_point = mean_point + self.pos
        body = model.body(f"obstacle_swall_{self.id}")
        body.pos = new_mean_point.tolist() + [1.0]

class Map:
    def __init__(self, obstacles: list[Obstacle] = None, extreme_points: np.ndarray = np.array([[0.0,0.0],[10.0,10.0]])):
        """
        Initializes the Map class.

        Args:
            obstacles (list[Obstacle], optional): A list of obstacles. Defaults to None.
            extreme_points (np.ndarray, optional): The extreme points of the map. Defaults to np.array([[0.0,0.0],[10.0,10.0]]).
        """
        if obstacles is None:
            obstacles = []
        self.obstacles = {}
        self.extreme_points = extreme_points
        for obs in obstacles:
            if not isinstance(obs, Obstacle):
                raise ValueError("All obstacles must be instances of Obstacle.")
            l = self.obstacles.setdefault(obs.__class__.__name__, [])
            l.append(obs)
        for k, v in self.obstacles.items():
            v.sort(key=lambda x: x.id)
    
    def get_obstacles(self, name: str) -> list[Obstacle]:
        """
        Returns the list of obstacles of a given type.

        Args:
            name (str): The name of the obstacle type.

        Returns:
            list[Obstacle]: The list of obstacles of the given type.
        """
        return self.obstacles.get(name, [])
    
    def dynamic_obstacles(self) -> list[DynamicObstacle]:
        """
        Returns the list of dynamic obstacles.

        Returns:
            list[DynamicObstacle]: The list of dynamic obstacles.
        """
        res = []
        for obs_list in self.obstacles.values():
            for obs in obs_list:
                if isinstance(obs, DynamicObstacle):
                    res.append(obs)
        return res
    
    def cost(self, pos: np.ndarray) -> float:
        """
        Computes the total cost at a given position due to all obstacles.

        Args:
            pos (np.ndarray): The position for which to compute the cost.

        Returns:
            float: The total computed cost.
        """
        c = 0.0
        for obs_list in self.obstacles.values():
            for obs in obs_list:
                c += obs.cost(pos)
        return c
    
    def add_to_spec(self, model_spec: dict):
        """
        Adds all obstacles to the model specification.

        Args:
            model_spec (dict): The model specification to which the obstacles are added.
        """
        for obs_list in self.obstacles.values():
            for obs in obs_list:
                print(f"Adding obstacle {obs} to model spec.")
                obs.add_to_spec(model_spec)
    
    def step(self, model: mujoco.MjModel, delta_t: float):
        """
        Advances the state of all dynamic obstacles by a given time step.

        Args:
            model (mujoco.MjModel): The MuJoCo model.
            delta_t (float): The time step by which to advance the obstacles' state.
        """
        for obs in self.dynamic_obstacles():
            obs.step(model, delta_t)
        