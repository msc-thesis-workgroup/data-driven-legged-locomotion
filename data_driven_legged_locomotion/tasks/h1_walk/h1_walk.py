import pathlib
import numpy as np

from data_driven_legged_locomotion.common import StateSpace, MujocoEnvironment

from pyquaternion import Quaternion

class H1WalkEnvironment(MujocoEnvironment):
    def __init__(self, n_samples = 1000):
        base_path = pathlib.Path(__file__).parent.parent.parent.parent
        print("[DEBUG] base_path: ", base_path)
        task_candidates = [path for path in pathlib.Path(base_path).rglob("mujoco_mpc-build/mjpc/tasks/h1/walk/task.xml")]
        print("[DEBUG] task candidates: ", task_candidates)
        if len(task_candidates) == 0:
            raise ValueError(f"Could not find h1_walk task in folder {base_path}, make sure to build mujoco_mpc")
        print("Task found: ", task_candidates[0])
        model_path = task_candidates[0]
        # upper_q_bounds = np.array([5.0,5.0,1.5, # Free joint linear position
        #                             1.,1.,1.,1., # Free joint angular position
        #                             0.43, 0.43, 2.53, 2.05, 0.52, 0.43, 0.43, 2.53, 2.05, 0.52, 2.35, 2.87, 3.11, 4.45, 2.61, 2.87, 0.34, 1.3, 2.61 # Rotoidal joints angular position
        #                             ])
        # lower_q_bounds = np.array([-5.0,-5.0,0., # Free joint linear position
        #                            -1.,-1.,-1.,-1., # Free joint angular position
        #                            -0.43, -0.43, -3.14, -0.26, -0.87, -0.43, -0.43, -3.14, -0.26, -0.87, -2.35, -2.87, -0.34, -1.3,  -1.25, -2.87, -3.11, -4.45, -1.25 # Rotoidal joints angular position
        #                            ])
        upper_q_bounds = np.ones(26)*100
        lower_q_bounds = -upper_q_bounds
        upper_dq_bounds = np.ones(6+19)*350
        lower_dq_bounds = -upper_dq_bounds
        upper_bounds = np.concatenate([upper_q_bounds, upper_dq_bounds])
        lower_bounds = np.concatenate([lower_q_bounds, lower_dq_bounds])
        deltas_q = (upper_q_bounds - lower_q_bounds) / n_samples
        deltas_dq = (upper_dq_bounds - lower_dq_bounds) / n_samples
        deltas = np.concatenate([deltas_q, deltas_dq])
        ss = StateSpace(26+25,
                        np.array(list(zip(lower_bounds, upper_bounds))),
                        deltas)
        super().__init__(ss, model_path)
        
def h1_walk_cost(x, k):
    r = np.array([10.0, 10.0])
    z_torso = 1.06
    #obs = np.array([4,4])
    costs = (x[:,0] - r[0])**2 + (x[:,1] - r[1])**2
    costs = np.squeeze(costs)
    costs += (x[:,2] - z_torso)**2
    return costs

def h1_walk_cost_trajectory(x, k):
    r = np.array([10.0, 0.0])
    costs = (x[0] - r[0])**2 + (x[1] - r[1])**2
    costs = np.squeeze(costs)
    return costs

class Cost:
    def __init__(self, obstacles_positions, obstacles_sizes):
        self.obstacles_positions = obstacles_positions
        self.obstacles_sizes = obstacles_sizes
        self.alpha = 1000.0
        self.beta = 10

    def get_cost_function(self):
        # def cost(x, k):
        #     r = np.array([10.0, 10.0])
        #     z_torso = 1.06
        #     costs = 20*(x[:,0] - r[0])**2 + (x[:,1] - r[1])**2 -10/(np.sqrt(( ((x[:,0]-r[0])/1000)**2 + ((x[:,1] - r[1])/1000)**2 +0.1)))
        #     costs = np.squeeze(costs)
        #     costs += self.alpha*(x[:,2] - z_torso)**2

        #     # obstacles are modeled as bivariate gaussian obstacles
        #     for i in range(len(self.obstacles_positions)):
        #         obs = self.obstacles_positions[i]
        #         size = self.beta*self.obstacles_sizes[i]
        #         costs += self.alpha*np.exp(-((x[:,0] - obs[0])**2/(2*size[0]**2) + (x[:,1] - obs[1])**2/(2*size[1]**2)))


        #     # Add wall costs. The walls are modeled as bivariate gaussian obstacles from the corners of the room: (-1,-1), (-1,11), (11,-1), (11,11).
        #     x_s = np.array([-1, 11])
        #     y_s = np.array([-1, 11])
        #     wall_size = 1
        #     wall_alpha = self.alpha
        #     wall_cost = 0
        #     for x_i in x_s:
        #         wall_cost += wall_alpha*np.exp(-((x[:,0] - x_i)**2/(2*wall_size**2)))
        #     for y_i in y_s:
        #         wall_cost += wall_alpha*np.exp(-((x[:,1] - y_i)**2/(2*wall_size**2)))
        #     costs += wall_cost
            
        #     return costs


        def cost(x, k):
            r = np.array([10.0, 10.0])
            z_torso = 0.96
            costs = 100*np.sqrt((x[0] - r[0])**2 + (x[1] - r[1])**2)
            if x[2] < 0.9:
                costs = float('inf')
                print("[WARNING] Torso too low. This policy could make the robot fall.")
                return costs

            # obstacles are modeled as bivariate gaussian obstacles
            for i in range(len(self.obstacles_positions)):
                obs = self.obstacles_positions[i]
                size = self.beta*self.obstacles_sizes[i]
                costs += self.alpha*np.exp(-((x[0] - obs[0])**2/(2*size[0]**2) + (x[1] - obs[1])**2/(2*size[1]**2)))

            # q1 = Quaternion(x[3:7])
            # q2 = Quaternion(np.array([1.0, 0.0, 0.0, 0.0]))
            # quat_cost = 50*Quaternion.absolute_distance(q1, q2)
            # print("[DEBUG] quat_cost: ", quat_cost,"other cost: ", costs)
            # costs += quat_cost

            return costs


        return cost

    # def get_cost_function(self):
    #     def cost(x, k):
    #         r = np.array([10.0, 0.0])
    #         z_torso = 0.96
    #         costs = 30*(x[:,0] - r[0])**4 + (x[:,1] - r[1])**4
    #         costs = np.squeeze(costs)
    #         costs += self.alpha*(x[:,2] - z_torso)**2
            
    #         return costs
        
    #     return cost