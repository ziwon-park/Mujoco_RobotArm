import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import time
import csv
 
from utils import compute
 
class RobotSimulator:
    def __init__(self, mjcf_path):
        self.model = load_model_from_path(mjcf_path)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.joint_names = ['Continuous_1', 'Continuous_2']
        self.joint_indices = [self.model.joint_names.index(name) for name in self.joint_names]
        self.angle = np.zeros(len(self.joint_indices))
 
    def compute_end_effector_positions(self):
       
        c1_range = np.linspace(-3.14, 3.14, 100)  
        c2_range = np.linspace(-3.14, 0, 50)     
        with open('end_effector_positions_only2.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["c1_angle", "c2_angle", "x", "y", "z"])
            for c1_angle in c1_range:
                for c2_angle in c2_range:
                    angles = [0]*7  
                    angles[0] = c1_angle  
                    angles[1] = c2_angle  
                    self.set_joint_angles(angles)
                    position = compute.compute_xc(self.sim.data.qpos)
                    writer.writerow([c1_angle, c2_angle, *position])
    
    def set_joint_angles(self, angles):
        self.sim.data.qpos[:len(angles)] = angles
        self.sim.forward()  
 
mjcf_path = "/home/robros/model_uncertainty/model/ROBROS/robot/base.xml"
robot_simulator = RobotSimulator(mjcf_path)
robot_simulator.compute_end_effector_positions()