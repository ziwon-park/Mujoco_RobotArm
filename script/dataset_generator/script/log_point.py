import os
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import mujoco
 
from utils import compute

class RobotSimulator:
    def __init__(self, mjcf_path, offset):
        self.model = load_model_from_path(mjcf_path)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.joint_names = ['Continuous_1', 'Continuous_2', 'Continuous_3',
                            'Continuous_4', 'Continuous_5', 'Continuous_6', 'Continuous_7']
        self.joint_indices = [self.model.joint_names.index(name) for name in self.joint_names]

        self.x_desired = [0,0,0]
        self.x_current = [0,0,0]
        self.Kp = 300
        self.Kd_joint = 0.0
        self.Ki = 0.1
        self.integral_error = np.zeros(3)
        self.Kd = 0.0
        self.prev_error = 0
        self.dt = 1.0 / 60.0  
        self.prev_angles = np.zeros(len(self.joint_indices))

        self.offset = offset

        self.rows = 3
        self.cols = 7

        self.q_prev = [0 for i in range(7)]
        self.marked_positions = []
    
    def write_marker(self, FK_position, Desired_position):
            self.viewer.add_marker(pos=(np.array([0,0,0])),
                                label="mujoco orientation", size=np.array([.01, .01, .01]))
            self.viewer.add_marker(pos=([-self.x_current[0], -self.x_current[1], self.x_current[2]]),
                                label=FK_position, size=np.array([.01, .01, .01]), rgba=np.array([1,0,0,1]), type=2)
            self.viewer.add_marker(pos=([-self.x_desired[0], -self.x_desired[1], self.x_desired[2]]),
                                label=Desired_position, size=np.array([.01, .01, .01]), rgba=np.array([0,1,0,1]), type=2)
            
            self.render_trajectory()
        
    def F_calcalator(self, angle):
        error = np.array(self.x_desired) - self.x_current
        self.integral_error += error * self.dt
        print("error is :", np.linalg.norm(error))

        angle_errors = angle - self.prev_angles
        tau_d = self.Kd_joint * angle_errors / self.dt
        self.prev_angles = angle
        derivative = (error - self.prev_error) / self.dt

        F = self.Kp * error + self.Ki * self.integral_error # PD, PI control
        # F = self.Kp * error # only PD control
        return F, error
    
    def add_marker(self, pos):
        # pos[2] = pos[2]+1.35
        position = [-pos[0], -pos[1], pos[2]]
        print("marker position is :", pos)
        self.viewer.add_marker(pos=position, size=np.array([.01, .01, .01]), rgba=np.array([0,1,0,1]), label="", type=2)

    def render_trajectory(self):
        """
        render all markers in marked_positions
        """
        for pos in self.marked_positions:
            # print("marked position length is : ",len(self.marked_positions))
            self.add_marker(pos)

    def move_robot(self, tau):
        for i, joint_idx in enumerate(self.joint_indices):
            self.sim.data.ctrl[joint_idx] = tau[i]

    def reset_simulation(self)

    def run_simulation(self):
        start_point = [0.3, -0.4, -0.3] 

        offset_length = 0.0001       
        num_points = 40    
        threshold = 0.1

        reached_start_point = False

        self.x_desired = start_point # desired point 초기화

        sim_state = self.sim.get_state()
            
        np.set_printoptions(precision=3)
 

        while True:

            print("desired point is :", self.x_desired)

            #### Get q positions

            angle = self.sim.data.qpos.copy()
        
            for i in range(7):
                angle[i] = angle[i] - self.offset[i]

            qr1, qr2, qr3, qr4, qr5, qr6, qr7 = angle

            #### Get end effector position

            self.x_current = compute.compute_xc(angle)

            #### PID controller

            F, error = self.F_calcalator(angle)

            if np.all(abs(self.prev_error - error) < 0.000001):
                reached_start_point = True


            self.prev_error = error  
            J = compute.compute_jacobian(angle, self.rows, self.cols)
            tau = J.T.dot(F) - 100*(angle - np.array(self.q_prev)) 

            #### Move Robot

            self.move_robot(tau)

            FK_position = (f"FK :{float(self.x_current[0]):.2f}, {float(self.x_current[1]):.2f}, {float(self.x_current[2]):.2f}")
            Desired_position = (f"Desired :{float(self.x_desired[0]):.2f}, {float(self.x_desired[1]):.2f}, {float(self.x_desired[2]):.2f}")

            #### Mujoco Markers for Visualization

            self.write_marker(FK_position, Desired_position)

            ####

            self.q_prev = angle
            self.sim.step()
            self.viewer.render()
            
 
mjcf_path = "/home/robros/model_uncertainty/model/ROBROS/mjmodel.xml"

offset = [0,0,0,0,0,0,0]


robot_sim = RobotSimulator(mjcf_path, offset)
robot_sim.run_simulation()