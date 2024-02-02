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
        # 이동 범위 설정
        c1_range = np.linspace(-3.14, 3.14, 100)  # Continuous_1 조인트 범위
        c2_range = np.linspace(-3.14, 0, 50)     # Continuous_2 조인트 범위
        # CSV 파일 초기화
        with open('end_effector_positions_only2.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["c1_angle", "c2_angle", "x", "y", "z"])
            for c1_angle in c1_range:
                for c2_angle in c2_range:
                    # 7개 조인트 각도 배열 생성, Continuous_1과 Continuous_2에 대해 설정, 나머지는 0 또는 기본값
                    angles = [0]*7  # 나머지 조인트는 0 또는 적절한 초기값으로 설정
                    angles[0] = c1_angle  # Continuous_1
                    angles[1] = c2_angle  # Continuous_2
                    self.set_joint_angles(angles)
                    position = compute.compute_xc(self.sim.data.qpos)
                    writer.writerow([c1_angle, c2_angle, *position])
    
    def set_joint_angles(self, angles):
        self.sim.data.qpos[:len(angles)] = angles
        self.sim.forward()  # 업데이트된 각도로 시뮬레이션 상태 업데이트
 
# 예제 사용
mjcf_path = "/home/robros/model_uncertainty/model/ROBROS/robot/base.xml"
robot_simulator = RobotSimulator(mjcf_path)
robot_simulator.compute_end_effector_positions()