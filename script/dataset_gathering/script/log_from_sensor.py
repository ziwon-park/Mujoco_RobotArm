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
 
# CSV 파일 설정
csv_filename = "../csv/joint_data_from_sensor.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    headers = ["Time"] + [f"{name}_pos" for name in joint_names] + [f"{name}_vel" for name in joint_names] + [f"{name}_acc" for name in joint_names]
    writer.writerow(headers)
 
    # Start Simulation
    start_time = time.time()
    while True:
        current_time = time.time() - start_time
        sim.step()
        viewer.render()

        # 관절 위치 변경
        for i in joint_indices:
            sim.data.qpos[i] += 0.001
 
        # 각 관절의 위치, 속도, 가속도 데이터 취득
        joint_positions = [sim.data.qpos[idx] for idx in joint_indices]
        joint_velocities = [sim.data.sensordata[sim.model.sensor_name2id(f"{name}_vel")] for name in joint_names]
        joint_accelerations = [sim.data.sensordata[sim.model.sensor_name2id(f"{name}_acc")] for name in joint_names]
 
        # 데이터 기록
        writer.writerow([current_time] + joint_positions + joint_velocities + joint_accelerations)
 
        # 시뮬레이션 종료 조건
        if current_time > 10:  # 10초 후에 종료
            break