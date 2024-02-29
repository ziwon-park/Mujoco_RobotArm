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
    def __init__(self, mjcf_path, offset, csv_path):
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

        self.desired_positions = self.load_desired_positions(csv_path)
        self.current_sequence_index = 0   

        self.collision_occurred = 0 
        self.collision_link_number = 0   

        self.cube_pos = [0,0,0]

        self.create_output_folder()
        self.reset_values()


    def randomize_cube_position(self):
        x_ranges = [(-0.3, -0.1), (0.1, 0.3)] 
        x_range = (-0.3, 0.3)
        y_range = (0.13, 0.3)
        z_range = (0.75, 1.15)
        
        selected_x_range = np.random.choice([0, 1])
        x_pos = np.random.uniform(*x_ranges[selected_x_range])
        
        # x_pos = np.random.uniform(*x_range)
        y_pos = np.random.uniform(*y_range)
        z_pos = np.random.uniform(*z_range)

        self.cube_pos = np.array([x_pos, y_pos, z_pos])

        cube_id = self.sim.model.body_name2id("cube")
        self.sim.model.body_pos[cube_id] = self.cube_pos

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

    def load_desired_positions(self, csv_path):
        """Load desired positions from a CSV file."""
        positions = []
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                positions.append([float(row[i]) for i in range(3)]) 
        return positions    

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

    def write_marker(self):
        FK_position = (f"FK :{float(self.x_current[0]):.2f}, {float(self.x_current[1]):.2f}, {float(self.x_current[2]):.2f}")
        Desired_position = (f"Desired :{float(self.x_desired[0]):.2f}, {float(self.x_desired[1]):.2f}, {float(self.x_desired[2]):.2f}")
        Cube_position = (f"Cube :{float(self.cube_pos[0]):.2f}, {float(self.cube_pos[1]):.2f}, {float(self.cube_pos[2]):.2f}")


        self.viewer.add_marker(pos=(np.array([0,0,0])),
                            label="mujoco orientation", size=np.array([.01, .01, .01]))
        self.viewer.add_marker(pos=([-self.x_current[0], -self.x_current[1], self.x_current[2]]),
                            label=FK_position, size=np.array([.02, .02, .02]), rgba=np.array([1,0,0,1]), type=2)
        self.viewer.add_marker(pos=([-self.x_desired[0], -self.x_desired[1], self.x_desired[2]]),
                            label=Desired_position, size=np.array([.02, .02, .02]), rgba=np.array([0,1,0,1]), type=2)
        self.viewer.add_marker(pos=([self.cube_pos[0], self.cube_pos[1], self.cube_pos[2]-1.35]),
                            label=Cube_position, size=np.array([.01, .01, .01]), rgba=np.array([0,0,1,1]), type=2)
        
        self.render_trajectory()
        
    def visualize_cube_area(self):
        x_ranges = [(-0.3, -0.1), (0.1, 0.3)]
        y_range = (0.13, 0.3)
        z_range = (0.75, 1.15)

        for x_range in x_ranges:
            for y in y_range:
                for z in z_range:
                    self.viewer.add_marker(label="", pos=np.array([x_range[0], y, z-1.35]), size=np.array([0.005, 0.005, 0.005]), rgba=np.array([0, 0, 1, 0.05]))
                    self.viewer.add_marker(label="", pos=np.array([x_range[1], y, z-1.35]), size=np.array([0.005, 0.005, 0.005]), rgba=np.array([0, 0, 1, 0.05]))

    
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

        self.move_robot(self.tau)

        self.visualize_cube_area()


    def reset_simulation(self):
        self.sim.reset()

    def reset_values(self):
        self.q_values = [[] for _ in range(7)]
        self.tau_values = [[] for _ in range(7)]
        self.collision_occurred = 0 
        self.collision_link_number = 0  

    def save_values(self):
        for i in range(7):
            filename = os.path.join(self.input_data_folder, f"fre_joint_{i+1}.csv")
            with open(filename, 'a', newline='') as file:
                writer = csv.writer(file)
                rounded_q_values = [round(val, 6) for val in self.q_values[i]]
                writer.writerow([self.collision_occurred, self.collision_link_number]+ rounded_q_values)

        for i in range(7):
            filename = os.path.join(self.target_data_folder, f"fre_joint_{i+1}.csv")
            with open(filename, 'a', newline='') as file:
                writer = csv.writer(file)
                rounded_tau_values = [round(val, 6) for val in self.tau_values[i]]
                writer.writerow([self.collision_occurred, self.collision_link_number]+ rounded_tau_values)


    def create_output_folder(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_folder = f"../csv/joint_data/collision_dataset/{current_time}"
        self.input_data_folder = os.path.join(base_folder, "input_data")
        self.target_data_folder = os.path.join(base_folder, "target_data")
        os.makedirs(self.input_data_folder, exist_ok=True)
        os.makedirs(self.target_data_folder, exist_ok=True)

    def run_simulation(self):
        # reached_point = False

        if self.current_sequence_index >= len(self.desired_positions):
            print("All sequences have been simulated.")
            return False  

        start_time = time.time()

        self.x_desired = self.desired_positions[self.current_sequence_index]  # Update desired position
        # self.x_desired = [0.15, -0.15, -0.65] 

        self.current_sequence_index += 1 
        self.reset_values()
        self.randomize_cube_position()


        while True: 
            current_time = time.time()

            if current_time - start_time > 1:
                # if not reached_point:
                #     # print("3 second passed")
                #     return False
                # else:
                break

            self.angle = self.sim.data.qpos.copy() 
            self.move_robot_to_position(self.x_desired)
            self.write_marker()
            self.check_collision()

        
            for i in range(7):
                self.angle[i] = self.angle[i] - self.offset[i]
                self.q_values[i].append(self.angle[i])
                self.tau_values[i].append(self.tau[i])

            self.q_prev = self.angle
            self.sim.step()
            self.viewer.render()

            # if np.linalg.norm(self.error) < 0.05:
            #     # print("done")
            #     reached_point = True
            #     self.reached_point_num = 1
            #     break 
        
        # return True  
            
            
mjcf_path = "/home/robros/model_uncertainty/model/ROBROS/robot/base.xml"
offset = [0,0,0,0,0,0,0]
csv_path = "/home/robros/model_uncertainty/script/visualize_workspace/double_random_selected_data.csv"

robot_sim = RobotSimulator(mjcf_path, offset,csv_path)

sequence_count = 1000

for sequence_number in tqdm(range(sequence_count), desc="Simulating Sequences"):
    result = robot_sim.run_simulation()
    robot_sim.save_values()
    # robot_sim.reset_simulation()