import numpy as np
import mujoco
from ..robots.robot import Robot

class PositionController():

    def __init__(self,robot: Robot, coeff: int = 1):
        #TODO: refactor this
        self.robot = robot
        self.kp = self.robot.get_kp()*coeff
        self.kd = self.robot.get_kd()*coeff
        self.nv = self.robot.get_nv()
        self.nq = self.robot.get_nq()

        self.qpos_index = [i for i in range(7,self.nq)] # Remove the free joint
        self.qvel_index = [i for i in range(6,self.nv)] # Remove the free joint
        # self.kp = np.array([200, 200, 200, 300, 40, 200, 200, 200, 300, 40, 300, 100, 100, 100, 100, 100, 100, 100, 100])
        # self.kd = np.array([5, 5, 5, 6, 2, 5, 5, 5, 6, 2, 6, 2, 2, 2, 2, 2, 2, 2, 2])

    def control_step(self,model, data, desired_q_pos, desired_q_vel) -> np.array:
        
        # compute controller data
        mass_matrix = np.ndarray(shape=(model.nv, model.nv), dtype=np.float64, order="C")
        mujoco.mj_fullM(model, mass_matrix, data.qM)
        mass_matrix = np.reshape(mass_matrix, (len(data.qvel), len(data.qvel)))
        self.mass_matrix = mass_matrix[self.qvel_index, :][:, self.qvel_index]
        #print(self.mass_matrix)
        #print(self.mass_matrix.shape)
        self.joint_pos = np.array(data.qpos[self.qpos_index])
        self.joint_vel = np.array(data.qvel[self.qvel_index])
        # print("joint_pos",self.joint_pos)
        # print("joint_vel",self.joint_vel)
        position_error = desired_q_pos - self.joint_pos
        vel_pos_error = desired_q_vel-self.joint_vel
        desired_torque = np.multiply(np.array(position_error), np.array(self.kp)) + np.multiply(vel_pos_error, self.kd)

        # Get torque_compensation
        self.torque_compensation = data.qfrc_bias[self.qvel_index]

        # Return desired torques plus gravity compensations
        self.torques = np.dot(self.mass_matrix, desired_torque) + self.torque_compensation

        return self.torques

    def control_step2(self,model, data, desired_q_pos, desired_q_vel=None) -> np.array:
        """In this version desired_q_vel is ignored, that means it is assumed to be zero"""
        kp = self.kp
        kd = self.kd

        actuator_length = data.qpos[7:self.nq] # self.data.actuator_length
        assert len(actuator_length) == len(desired_q_pos)
        error = desired_q_pos - actuator_length
        m = model
        d = data

        empty_array = np.zeros(m.actuator_dyntype.shape)

        ctrl_dot = np.zeros(m.actuator_dyntype.shape) if np.array_equal(m.actuator_dyntype,empty_array) else d.act_dot[m.actuator_actadr + m.actuator_actnum - 1]

        error_dot = ctrl_dot - data.qvel[6:self.nv]
        assert len(error_dot) == len(error)

        joint_torques = kp*error + kd*error_dot

        return joint_torques

if __name__ == "__main__":

    PATH_TO_MODEL = "h1/scene.xml"

    traj = np.load("./traj.npy")
    print(traj.shape)

    # Load the model
    model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
    data = mujoco.MjData(model)

    controller = PositionController()
    steps = 5
    for i in range(steps):
        target_qpos = traj[i][1:27]
        target_qvel = traj[i][27:52]
        assert len(target_qpos) == model.nq
        assert len(target_qvel) == model.nv
        #print("target_qpos", target_qpos)

        torques = controller.control_step(model,data,target_qpos[7:26],target_qvel[6:25])
        #print("torques",torques)

        data.ctrl[:] = torques
        mujoco.mj_step(model,data)

        print("Pos error:", target_qpos[:26] - data.qpos[:26])