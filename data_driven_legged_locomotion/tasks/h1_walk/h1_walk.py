import pathlib
import numpy as np

from data_driven_legged_locomotion.common import StateSpace, MujocoEnvironment

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
    r = np.array([10.0, 0.0])
    costs = (x[:,0] - r[0])**2 + (x[:,1] - r[1])**2
    costs = np.squeeze(costs)
    return costs



def h1_walk_cost_trajectory(states, k):
    r = np.array([10.0, 10.0])
    obstacle = np.array([4.0, 4.0])
    cost = 0

    # length = states.shape[0]
    # alpha = 1/length
    # for i in range(length):
    #     cost += alpha*((states[i,0] - r[0])**2 + (states[i,1] - r[1])**2)
    #     alpha = alpha + 1/length
    
    cost = (states[-1,0] - r[0])**2 + (states[-1,1] - r[1])**2

    # create a gaussian obstacle centered at obstacle with variance var
    var = 1.5
    coeff = 20
    cost += coeff*np.exp(-((states[-1,0] - obstacle[0])**2 + (states[-1,1] - obstacle[1])**2)/(2*var))

    return cost