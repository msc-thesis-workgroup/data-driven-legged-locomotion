from abc import ABC, abstractmethod
import functools
import mujoco
import numpy as np
import scipy

class Obstacle(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def cost(self, pos: np.ndarray) -> float:
        raise NotImplementedError("Method not implemented.")
    
    @abstractmethod
    def add_to_spec(self, model_spec: dict):
        raise NotImplementedError("Method not implemented.")
    
class DynamicObstacle(Obstacle):
    def __init__(self):
        self.transition_end = False
    
    @abstractmethod
    def step(self, delta_t: float):
        raise NotImplementedError("Method not implemented.")

class Cylinder(Obstacle):
    last_id = 0
    
    def __init__(self, pos: np.ndarray, radius: float = 0.8, height: float = 0.5):
        super().__init__()
        self.id = Cylinder.last_id
        Cylinder.last_id += 1
        self.radius = radius
        self.height = height
        self.pos = pos

    def cost(self, pos: np.ndarray) -> float:
        return 30*scipy.stats.multivariate_normal.pdf(pos, mean=self.pos, cov=0.4*np.eye(2))
    
    def add_to_spec(self, model_spec: dict):
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
        super().__init__()
        self.id = SlidingWall.last_id
        SlidingWall.last_id += 1
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.shift_vect = shift_vect
        self.width = width
        self.height = height
        self.length = np.linalg.norm(end_pos-start_pos)
        self.yaw = np.arctan2(end_pos[1]-start_pos[1], end_pos[0]-start_pos[0])
        self.mean_point = (start_pos + end_pos)/2
        self.pos = np.zeros(2) # Position of the center of the obstacle
        
    def cost(self, pos: np.ndarray) -> float:
        n_obs = max(1,int(self.length / 1.0))
        obs_perc = np.linspace(0, 1, n_obs)
        start_point, end_point = self.start_pos, self.end_pos
        if self.transition_end:
            start_point = start_point + self.shift_vect
            end_point = end_point + self.shift_vect
        c = 0.0
        for perc in obs_perc:
            obs_point = start_point + perc * (end_point - start_point)
            c += 100*scipy.stats.multivariate_normal.pdf(pos, mean=obs_point, cov=0.2*np.eye(2))
        return c
    
    def add_to_spec(self, model_spec: dict):
        body = model_spec.worldbody.add_body()
        body.name = f"obstacle_swall_{self.id}"
        body.pos = self.mean_point.tolist() + [self.height]
        body.quat = scipy.spatial.transform.Rotation.from_euler('z', self.yaw).as_quat(scalar_first=True)
        geom = body.add_geom()
        geom.name = f"obstacle_swall_geom_{self.id}"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [self.length/2, self.width, self.height]
        geom.rgba = [1, 0, 0, 1]
        
    def step(self, model: mujoco.MjModel, delta_t: float):
        if self.transition_end:
            return
        translation = np.linalg.norm(self.shift_vect)
        direction = self.shift_vect/translation
        if np.dot(direction, self.pos) >= translation:
            self.transition_end = True
            return
        self.pos += direction * delta_t * 10.0
        start_point, end_point = self.start_pos, self.end_pos
        mean_point = (start_point + end_point)/2
        new_mean_point = mean_point + self.pos
        body = model.body(f"obstacle_swall_{self.id}")
        body.pos = new_mean_point.tolist() + [1.0]

class Map:
    def __init__(self, obstacles: list[Obstacle] = None, extreme_points: np.ndarray = np.array([[0.0,0.0],[10.0,10.0]])):
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
        return self.obstacles.get(name, [])
    
    def dynamic_obstacles(self):
        res = []
        for obs_list in self.obstacles.values():
            for obs in obs_list:
                if isinstance(obs, DynamicObstacle):
                    res.append(obs)
        return res
    
    def cost(self, pos: np.ndarray) -> float:
        c = 0.0
        for obs_list in self.obstacles.values():
            for obs in obs_list:
                c += obs.cost(pos)
        return c
    
    def add_to_spec(self, model_spec: dict):
        for obs_list in self.obstacles.values():
            for obs in obs_list:
                print(f"Adding obstacle {obs} to model spec.")
                obs.add_to_spec(model_spec)
    
    def step(self, model: mujoco.MjModel, delta_t: float):
        for obs in self.dynamic_obstacles():
            obs.step(model, delta_t)
        