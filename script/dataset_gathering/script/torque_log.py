#!/usr/bin/env python3

# Free motion 상태에서 모터 토크 값을 csv 파일로 저장하는 코드

import os
from mujoco_py import load_model_from_path, MjSim, MjViewer
import csv
import time
 
model = load_model_from_path("/home/robros/model_uncertainty/model/gen3lite.xml")
 
# Simulation Setup
sim = MjSim(model)
viewer = MjViewer(sim)
 
joint_names = ['J0', 'J1', 'J2', 'J3', 'J4']
joint_indices = [model.joint_names.index(name) for name in joint_names]

desired_joint_index = model.joint_names.index(joint_names[0])
 
# CSV 파일 설정
csv_filename = "joint_velocities.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time"] + joint_names)
 
    # Start Simulation
    start_time = time.time()
    while True:
        current_time = time.time() - start_time
        sim.step()
        viewer.render()

        for i in joint_indices:
            sim.data.qpos[i] += 0.001

        # sim.data.qpos[desired_joint_index] += 0.001
 
        # 각 관절의 속도(qvel) 기록
        joint_velocities = [sim.data.qvel[idx] for idx in joint_indices]
        writer.writerow([current_time] + joint_velocities)
 
        # # 시뮬레이션 종료 조건 (예: 특정 시간 후에 종료)
        # if current_time > 10:  # 10초 후에 종료
        #     break