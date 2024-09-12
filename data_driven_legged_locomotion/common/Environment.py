import mujoco
import numpy as np
from pathlib import Path
from .StateSpace import StateSpace
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET


class MujocoEnvironment:
    def __init__(self, ss: StateSpace, model_path: Path) -> None:
        self.ss = ss
        self.model_path = model_path
        model_path = str(model_path)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.n_states = self.model.nq + self.model.nv
        self.n_actions = self.model.nu
        if self.n_states != ss.n_states:
            raise ValueError(f"State space dimension {ss.n_states} does not match model dimension {ss.n_states}")
        self.data = mujoco.MjData(self.model)
    
    @property
    def timestep(self):
        return self.model.opt.timestep
    
    @property
    def time(self):
        return self.data.time
    
    def get_state(self, split=False) -> np.ndarray:
        qpos = self.data.qpos[:]
        qvel = self.data.qvel[:]
        if split:
            return qpos, qvel
        return np.concatenate((qpos, qvel))
     
    def step(self, u: np.ndarray|float):
        if isinstance(u, float):
            u = np.array([u])
        if u.shape != (self.n_actions,):
            raise ValueError(f"Action dimension {u.shape} does not match model dimension {self.n_actions}")
        self.data.ctrl = u
        mujoco.mj_step(self.model, self.data)
    
    def reset(self):
        mujoco.mj_reset(self.model, self.data)
        
    def launch_passive_viewer(self):
        return mujoco.viewer.launch_passive(self.model, self.data)
    


    def create_obstacles(self, obstacle_positions, obstacle_sizes, obstacle_rgba=None):
        """
        Dynamically adds obstacles to the MJCF (XML) model based on positions and sizes.
        
        Args:
            xml_path (str): Path to the original XML file (model without obstacles).
            new_xml_path (str): Path to save the new XML file (model with obstacles).
            obstacle_positions (list of list): A list of [x, y, z] positions for each obstacle.
            obstacle_sizes (list of list): A list of [x_size, y_size, z_size] for each obstacle.
            obstacle_rgba (list of list): Optional list of RGBA colors for each obstacle.
                                        Default is red for all obstacles.
        """
        # Load the original XML file
        tree = ET.parse(str(self.model_path))
        root = tree.getroot()

        new_xml_path = str(self.model_path.parent / "new_model.xml")
        # Find the <worldbody> element to insert new obstacles
        worldbody = root.find('worldbody')

        # Set default RGBA colors (if not provided)
        if obstacle_rgba is None:
            obstacle_rgba = [[1, 0, 0, 1] for _ in range(len(obstacle_positions))]

        # Loop through each obstacle to add
        for i, (pos, size, rgba) in enumerate(zip(obstacle_positions, obstacle_sizes, obstacle_rgba)):
            # Create a new body for the obstacle
            obstacle_body = ET.Element('body', name=f'obstacle_{i}', pos=" ".join(map(str, pos)))

            # Create the geometry for the obstacle (we'll use a cylinder)
            geom = ET.SubElement(obstacle_body, 'geom', name=f'obstacle_geom_{i}', type="cylinder",
                                size=" ".join(map(str, size)), rgba=" ".join(map(str, rgba)))

            # Append the new obstacle to the <worldbody>
            worldbody.append(obstacle_body)

        # Save the modified XML to the new path
        tree.write(new_xml_path)

        print(f"Obstacles added and new XML saved to {new_xml_path}")

        # Update the model with the new XML
        self.model = mujoco.MjModel.from_xml_path(new_xml_path)
        self.data = mujoco.MjData(self.model)

        
    def reach_state(self, desired_state: np.ndarray, n_steps: int, viewer, video, renderer, frame_count, video_fps):
        """Returns the control action to reach the desired state at the given time."""
        assert desired_state.shape[0] == self.n_states
        
        print("n_steps: ", n_steps)
        print("n_steps*self.timestep: ", n_steps*self.timestep)
        target_pos = desired_state[0:26]
        target_vel = desired_state[26:51]
             
        starting_pos = self.data.qpos.copy()
        starting_vel = self.data.qvel.copy()
        interpolator = TrajectoryInterpolator(starting_pos[7:26], starting_vel[6:25], n_steps*self.timestep, target_pos[7:26], target_vel[6:25])
        interpolator2 = TrajectoryInterpolator(starting_pos[0:3], starting_vel[0:3], n_steps*self.timestep, target_pos[0:3], target_vel[0:3])
        
        target_rotation = R.from_quat(target_pos[3:7].tolist(), scalar_first=True)
        target_angles = target_rotation.as_euler('xyz', degrees=False)

        starting_rotation = R.from_quat(starting_pos[3:7].tolist(), scalar_first=True)
        starting_angles = starting_rotation.as_euler('xyz', degrees=False)
            
        interpolator3 = TrajectoryInterpolator(starting_angles, starting_vel[3:6], n_steps*self.timestep, target_angles, target_vel[3:6])
        
        #spline = RotationSpline([0,n_steps*self.timestep], [R.from_quat(starting_pos[3:7]).as_quat(scalar_first=True), R.from_quat(target_pos[3:7]).as_quat(scalar_first=True)])
        time = 0

        for i in range(n_steps):

            #
            self.data.qfrc_applied = np.zeros_like(self.data.qfrc_applied)
            self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)
            self.data.ctrl = np.zeros_like(self.data.ctrl)
            #

            old_acc = self.data.qacc.copy()

            curr_pos = self.data.qpos.copy()
            curr_vel = self.data.qvel.copy()

            # self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)
            # self.data.qfrc_applied = np.zeros_like(self.data.qfrc_applied)

            time += self.timestep
            self.data.qacc[6:25] = interpolator.get_acc(time)
            
            self.data.qacc[0:3] = interpolator2.get_acc(time)
            self.data.qacc[3:6] = interpolator3.get_acc(time)
            #mujoco.mj_forward(self.model, self.data)
            mujoco.mj_inverse(self.model, self.data)

            self.data.qacc = old_acc
            sol = self.data.qfrc_inverse.copy()
            
            # u = sol[6:25]
            # self.data.qfrc_applied[0:6] = sol[0:6]
            # # if an element is greater than the ctrlrange, print it
            # for i in range(len(u)):
            #     if abs(u[i]) > self.model.actuator_ctrlrange[i][1]:
            #         print(f"Control {i} is greater than ctrlrange: {u[i]}")

            # self.data.ctrl = u
            self.data.qfrc_applied = sol
            mujoco.mj_step(self.model, self.data)
        
            viewer.sync()

            if frame_count < self.data.time * video_fps:
                renderer.update_scene(self.data, camera="top")
                pixels = renderer.render()
                video.add_image(pixels)
                frame_count += 1
        
        print("Position Error: ", self.data.qpos - desired_state[:self.model.nq])
        print("Velocity Error: ", self.data.qvel - desired_state[self.model.nq:])
        
        return frame_count


class TrajectoryInterpolator:

    def __init__(self, starting_qpos, starting_qvel , duration ,final_qpos, final_qvel, time=0.0):
        self.starting_qpos = starting_qpos
        self.starting_qvel = starting_qvel
        self.duration = duration
        self.final_qpos = final_qpos
        self.final_qvel = final_qvel
        self.time = time

        # Setup the interpolator
        self.setup()

    def setup(self):
        self.a0 = self.starting_qpos
        self.a1 = self.starting_qvel
        self.a2 = (3*(self.final_qpos - self.starting_qpos) - (2*self.starting_qvel + self.final_qvel)*self.duration) / (self.duration**2)
        self.a3 = (2*(self.starting_qpos - self.final_qpos) + (self.starting_qvel + self.final_qvel)*self.duration) / (self.duration**3)


    def get_acc(self, t):
        """Compute the desired position at time t"""
        # if round(t,5) > self.time+self.duration:
        #     print("WARNING: Time is greater than the duration of the trajectory")
        #     print("t: ", t, "duration: ", self.duration, "time_step: ", self.time_step)
        #     return np.zeros(19)
        #t = np.round(t - self.time,5)
        
        t = np.round(t,5)
        return 6*self.a3*t + 2*self.a2
    