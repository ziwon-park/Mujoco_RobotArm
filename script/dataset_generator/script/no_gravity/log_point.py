import os
import shutil
from tqdm import tqdm

import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import mujoco
import time, datetime
import csv
 
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
        self.error = 0
        self.prev_error = 0
        self.dt = 1.0 / 60.0  
        self.angle = np.zeros(len(self.joint_indices))
        self.prev_angles = np.zeros(len(self.joint_indices))

        self.offset = offset

        self.rows = 3
        self.cols = 7

        self.q_prev = [0 for i in range(7)]
        self.tau = None
        self.tau_values = None
        self.q_values = None
        self.marked_positions = []

        self.collision_occurred = 0 
        self.collision_link_number = 0 

        self.create_output_folder()
        self.reset_values()

    def randomize_cube_position(self):
        x_ranges = [(-0.2, -0.1), (0.1, 0.2)] 
        x_range = (0.15, 0.35)
        y_range = (0.16, 0.19)
        z_range = (0.75, 1.15)
        
        selected_x_range = np.random.choice([0, 1])
        x_pos = np.random.uniform(*x_ranges[selected_x_range])
        
        x_pos = np.random.uniform(*x_range)
        y_pos = np.random.uniform(*y_range)
        z_pos = np.random.uniform(*z_range)

        cube_pos = np.array([x_pos, y_pos, z_pos])

        cube_id = self.sim.model.body_name2id("cube")
        self.sim.model.body_pos[cube_id] = cube_pos

    def check_collision(self):
        cube_geom_id = self.sim.model.geom_name2id("cube") 

        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if contact.geom1 == cube_geom_id or contact.geom2 == cube_geom_id:
                other_geom_id = contact.geom2 if contact.geom1 == cube_geom_id else contact.geom1
                other_geom_name = self.sim.model.geom_id2name(other_geom_id)
                if "link_col_" in other_geom_name:
                    self.collision_occurred = 1  
                    self.collision_link_number = int(other_geom_name.split('_')[2])  # 'link_' 다음에 오는 번호 추출
                    break  

        return self.collision_occurred, self.collision_link_number


    
    def write_marker(self):
        FK_position = (f"FK :{float(self.x_current[0]):.2f}, {float(self.x_current[1]):.2f}, {float(self.x_current[2]):.2f}")
        Desired_position = (f"Desired :{float(self.x_desired[0]):.2f}, {float(self.x_desired[1]):.2f}, {float(self.x_desired[2]):.2f}")

        self.viewer.add_marker(pos=(np.array([0,0,0])),
                            label="mujoco orientation", size=np.array([.01, .01, .01]))
        self.viewer.add_marker(pos=([-self.x_current[0], -self.x_current[1], self.x_current[2]]),
                            label=FK_position, size=np.array([.01, .01, .01]), rgba=np.array([1,0,0,1]), type=2)
        self.viewer.add_marker(pos=([-self.x_desired[0], -self.x_desired[1], self.x_desired[2]]),
                            label=Desired_position, size=np.array([.01, .01, .01]), rgba=np.array([0,1,0,1]), type=2)
        
        self.render_trajectory()
        
    def F_calcalator(self, angle):
        self.error = np.array(self.x_desired) - self.x_current
        self.integral_error += self.error * self.dt
        # print("error is :", np.linalg.norm(self.error))

        angle_errors = angle - self.prev_angles
        tau_d = self.Kd_joint * angle_errors / self.dt
        self.prev_angles = angle
        derivative = (self.error - self.prev_error) / self.dt

        F = self.Kp * self.error + self.Ki * self.integral_error # PD, PI control
        # F = self.Kp * error # only PD control
        return F
    
    def add_marker(self, pos):
        position = [-pos[0], -pos[1], pos[2]]
        # print("marker position is :", pos)
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

    def move_robot_to_position(self, position):
        self.angle = self.sim.data.qpos.copy()
        self.x_current = compute.compute_xc(self.angle)
        F = self.F_calcalator(self.angle)
        self.prev_error = self.error  
        J = compute.compute_jacobian(self.angle, self.rows, self.cols)
        self.tau = J.T.dot(F) - 100*(self.angle - np.array(self.q_prev)) 
        # self.save_tau_values(tau)

        self.move_robot(self.tau)
        self.write_marker()


    def reset_simulation(self):
        self.sim.reset()

    def reset_values(self):
        self.q_values = [[] for _ in range(7)]
        self.tau_values = [[] for _ in range(7)]

    def save_values(self):
        for i in range(7):
            filename = os.path.join(self.input_data_folder, f"fre_joint_{i+1}.csv")
            with open(filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.collision_occurred, self.collision_link_number]+self.q_values[i])

        for i in range(7):
            filename = os.path.join(self.target_data_folder, f"fre_joint_{i+1}.csv")
            with open(filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.tau_values[i])


    def create_output_folder(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_folder = f"../csv/joint_data/temp/{current_time}"
        self.input_data_folder = os.path.join(base_folder, "input_data")
        self.target_data_folder = os.path.join(base_folder, "target_data")
        os.makedirs(self.input_data_folder, exist_ok=True)
        os.makedirs(self.target_data_folder, exist_ok=True)

    def run_simulation(self):
        reached_point = False
        start_time = time.time()

        start_point = [0.15, -0.15, -0.65] 

        self.x_desired = start_point # desired point 초기화

        update_count = 0
        max_updates = 50

        self.reset_values()
        self.randomize_cube_position()

        while reached_point is False: # 초기 위치로 이동 
            self.angle = self.sim.data.qpos.copy()
            self.move_robot_to_position(self.x_desired)

            self.q_prev = self.angle
            self.sim.step()
            self.viewer.render()

            if np.linalg.norm(self.error) < 0.06:
                # print("near enough")
                reached_point = True


        while True: 
            current_time = time.time()
            np.set_printoptions(precision=3)

            # print("desired point is :", self.x_desired)

            if current_time - start_time > 3:
                if not reached_point:
                    print("5 second passed")
                    return False
                else:
                    break

            self.angle = self.sim.data.qpos.copy() 
        
            for i in range(7):
                self.angle[i] = self.angle[i] - self.offset[i]
                self.q_values[i].append(self.angle[i])
                self.tau_values[i].append(self.tau[i])

            self.move_robot_to_position(self.x_desired)
            self.check_collision()

            self.q_prev = self.angle
            self.sim.step()
            self.viewer.render()

            #### 
            if np.linalg.norm(self.error) < 0.06:
                # print("near enough (not initial loop)")
                reached_point = True
                    
                if update_count < max_updates:
                    self.x_desired[0] = self.x_desired[0] - 0.01
                    update_count += 1
                else:
                    return False
            # return True
            
            
mjcf_path = "/home/robros/model_uncertainty/model/ROBROS/robot/base.xml"
offset = [0,0,0,0,0,0,0]

robot_sim = RobotSimulator(mjcf_path, offset)

sequence_count = 50

for sequence_number in tqdm(range(sequence_count), desc="Simulating Sequences"):
    robot_sim.run_simulation()
    robot_sim.save_values()
    robot_sim.reset_simulation()


# for sequence_number in range(sequence_count):
#     robot_sim.run_simulation()
#     robot_sim.save_values()
#     robot_sim.reset_simulation()