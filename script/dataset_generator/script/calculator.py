import csv
 
def calculate_velocity_acceleration(input_filename, output_filename):
    with open(input_filename, 'r') as infile, open(output_filename, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
 
        headers = next(reader)  # 첫 번째 행 (헤더) 읽기
        num_joints = len(headers) - 1  # 시간 열 제외
 
        # 새로운 헤더 작성 (속도와 가속도 추가)
        new_headers = headers + [f"V_{i}" for i in range(num_joints)] + [f"A_{i}" for i in range(num_joints)]
        writer.writerow(new_headers)
 
        # 초기 데이터 읽기
        prev_row = next(reader)
        prev_time = float(prev_row[0])
        prev_pos = [float(x) for x in prev_row[1:]]
 
        for row in reader:
            current_time = float(row[0])
            current_pos = [float(x) for x in row[1:]]
            dt = current_time - prev_time
            velocities = [(cp - pp) / dt for cp, pp in zip(current_pos, prev_pos)]
 
            # 가속도 계산 (속도의 변화율)
            if 'prev_velocities' in locals():
                accelerations = [(v - pv) / dt for v, pv in zip(velocities, prev_velocities)]
            else:
                accelerations = [0] * num_joints  # 첫 번째 행의 가속도는 0으로 설정
 
            writer.writerow(row + velocities + accelerations)
 
            prev_time, prev_pos, prev_velocities = current_time, current_pos, velocities
 
 
# 파일 이름
input_filename = "/home/robros/model_uncertainty/script/dataset_gathering/csv/joint_data.csv"
output_filename = "/home/robros/model_uncertainty/script/dataset_gathering/csv/test.csv"
 
# 가속도 계산 및 파일 쓰기
calculate_velocity_acceleration(input_filename, output_filename)