import os
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import mujoco
import math
import torch
import pytorch_kinematics as pk

# Load Model
urdf_path = "/home/robros/model_uncertainty/model/gen3lite.urdf"
mjcf_path = "/home/robros/model_uncertainty/model/gen3lite.xml"

# urdf_path = "/home/robros/anaconda3/envs/mujoco_py/lib/python3.8/site-packages/pybullet_data/franka_panda/panda.urdf"
# mjcf_path = "/home/robros/Downloads/robosuite-master/robosuite/models/assets/robots/kinova3/robot.xml"

model = load_model_from_path(mjcf_path)
chain = pk.build_serial_chain_from_urdf(open(urdf_path).read(), "UPPER_WRIST")

sim = MjSim(model)
viewer = MjViewer(sim)

joint_names = ['J0', 'J1', 'J2', 'J3', 'J4']

joint_indices = [model.joint_names.index(name) for name in joint_names]
x_desired = [-0.25, -0.25, -0.4]

# PD parameters
Kp = 5000
Kd_joint = 0.05
Kd = 0.0
prev_error = 0  
dt = 1.0 / 60.0  # Assuming 60 Hz simulation frequency

prev_angles = np.zeros(len(joint_indices))

while True:
    x_current = sim.data.get_body_xipos("UPPER_WRIST")
    angle = torch.Tensor(sim.data.qpos.copy())

    print(angle)
    
    print("x_desired:", x_desired)
    print("x_current:", x_current)

    # PD control
    error = x_desired - x_current
    angle_errors = angle - prev_angles
    tau_d = Kd_joint * angle_errors / dt
    prev_angles = angle
    derivative = (error - prev_error) / dt
    F = Kp * error + Kd * derivative
    prev_error = error  


    if np.linalg.norm(error) < 0.01:  # Threshold for stopping
        print("break")
        break

    J = chain.jacobian(angle)
    J = J.squeeze(0).numpy()

    new_J = J[0:3, :] 
    print("new J :", new_J)

    # gravity_compensation = np.array([0.0, 0.0, 9.81]) * model.body_mass[0] 
 
    # tau = new_J.T.dot(F) + tau_d
    tau = new_J.T.dot(F)
    print("tau:",tau)

    # tau = 10000*tau
    # print(tau)
    # tau = new_J.T.dot(F - gravity_compensation)

    for i, joint_idx in enumerate(joint_indices):
        print("joint_idx :", joint_idx)
        sim.data.ctrl[joint_idx] =  tau[i]

    sim.step()
    viewer.render()