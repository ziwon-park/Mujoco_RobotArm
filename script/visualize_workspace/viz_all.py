import numpy as np
from mujoco_py import load_model_from_path, MjSim
import csv
import itertools
from tqdm import tqdm  # tqdm 라이브러리 import

from utils import compute

class RobotSimulator:
    def __init__(self, mjcf_path):
        self.model = load_model_from_path(mjcf_path)
        self.sim = MjSim(self.model)
        self.joint_names = ['Continuous_1', 'Continuous_2', 'Continuous_3', 
                            'Continuous_4', 'Continuous_5', 'Continuous_6', 'Continuous_7']
        self.joint_indices = [self.model.joint_names.index(name) for name in self.joint_names]
        original_angle_ranges = [
            (-2.09, 1.57),
            (-2.09, 0),
            (-1.57, 1.57),
            (0, 2.09),
            (-1.57, 1.57),
            (-1.04, 1.04),
            (-1.04, 1.04),
        ]
        self.angle_ranges = [
            np.linspace(start, end, num=self.calculate_num_values(start, end))
            for start, end in original_angle_ranges
        ]
 
    def calculate_num_values(self, range_min, range_max, min_points=3, max_points=20):
        range_size = range_max - range_min
        full_range_size = 4.18
        num = int(min_points + (range_size / full_range_size) * (max_points - min_points))-3
        return max(min_points, min(num, max_points))
 
    def compute_all_positions(self):
        total_combinations = np.product([len(range_) for range_ in self.angle_ranges])
        with open('end_effector_positions.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["joint_angles"] + ["x", "y", "z"])
            # tqdm 사용하여 진행률 표시
            for angles in tqdm(itertools.product(*self.angle_ranges), total=total_combinations, desc="Computing positions"):
                self.set_joint_angles(angles)
                position = compute.compute_xc(self.sim.data.qpos)
                writer.writerow([angles] + list(position))
 
    def set_joint_angles(self, angles):
        for i, angle in enumerate(angles):
            self.sim.data.qpos[self.joint_indices[i]] = angle
        self.sim.forward()

# 예제 사용
mjcf_path = "/home/robros/model_uncertainty/model/ROBROS/robot/base.xml"
robot_simulator = RobotSimulator(mjcf_path)
robot_simulator.compute_all_positions()
