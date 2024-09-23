import numpy as np

class TrajectoryPlanner:
    """This class is used to plan a trajectory from a starting position to a final position using a quintic polynomial for the position"""

    def __init__(self, starting_qpos, starting_qvel , duration ,final_qpos, final_qvel):
        self.starting_qpos = starting_qpos
        self.starting_qvel = starting_qvel
        self.duration = duration
        self.final_qpos = final_qpos
        self.final_qvel = final_qvel

        # Setup the interpolator
        self.setup()


    def setup(self):
        self.a0 = self.starting_qpos
        self.a1 = self.starting_qvel
        self.a2 = (3*(self.final_qpos - self.starting_qpos) - (2*self.starting_qvel + self.final_qvel)*self.duration) / (self.duration**2)
        self.a3 = (2*(self.starting_qpos - self.final_qpos) + (self.starting_qvel + self.final_qvel)*self.duration) / (self.duration**3)


    def get_acc(self, t):
        """Compute the desired position at time t"""
        t = round(t, 4) # round to avoid floating point errors. WARNING: This may not be the best solution
        if t > self.duration:
            return np.zeros_like(self.final_qvel)

        return 6*self.a3*t + 2*self.a2
    
    def get_pos(self, t):
        """Compute the desired position at time t"""

        t = round(t, 4)
        if t > self.duration:
            return np.zeros_like(self.final_qpos)

        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3
    
    def get_vel(self, t):
        """Compute the desired velocity at time t"""
        t = round(t, 4)
        if t > self.duration:
            return np.zeros_like(self.final_qvel)

        return self.a1 + 2*self.a2*t + 3*self.a3*t**2
        


if __name__ == "__main__":

    qpos_start = np.array([1,1])
    qpos_end = np.array([2,2])

    qvel_start = np.array([1,1])
    qvel_end = np.array([2,2])

    duration = 1.0
    trajectory_interpolator = TrajectoryPlanner(qpos_start, qvel_start, duration, qpos_end, qvel_end)

    print("positions test")
    print("t = 0.0, pos =", trajectory_interpolator.get_pos(0.0))
    print("t = 0.5, pos =", trajectory_interpolator.get_pos(0.5))
    print("t = 1.0, pos =", trajectory_interpolator.get_pos(1.0))
    print("t = 1.5, pos =", trajectory_interpolator.get_pos(1.5))

    print("accelerations test")
    print("t = 0.0, acc =", trajectory_interpolator.get_acc(0.0))
    print("t = 0.5, acc =", trajectory_interpolator.get_acc(0.5))
    print("t = 1.0, acc =", trajectory_interpolator.get_acc(1.0))
    print("t = 1.5, acc =", trajectory_interpolator.get_acc(1.5))



    print("velocities test")
    print("t = 0.0, vel =", trajectory_interpolator.get_vel(0.0))
    print("t = 0.5, vel =", trajectory_interpolator.get_vel(0.5))
    print("t = 1.0, vel =", trajectory_interpolator.get_vel(1.0))
    print("t = 1.5, vel =", trajectory_interpolator.get_vel(1.5))
    
    

    
