import csv
 
# 파일 읽기 및 가속도 계산
def calculate_joint_accelerations(input_filename, output_filename):
    with open(input_filename, mode='r') as infile, open(output_filename, mode='w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
 
        # 헤더 읽기
        headers = next(reader)
        writer.writerow(headers + ["A_" + h for h in headers[1:]])
 
        # 이전 단계의 데이터 초기화
        prev_time, prev_velocities = 0, []
        for row in reader:
            current_time = float(row[0])
            current_velocities = [float(v) for v in row[1:]]
 
            # 가속도 계산 (첫 번째 행 제외)
            if prev_velocities:
                dt = current_time - prev_time
                accelerations = [(curr - prev) / dt for curr, prev in zip(current_velocities, prev_velocities)]
                writer.writerow(row + accelerations)
 
            prev_time, current_time
            prev_velocities = current_velocities
 
# 파일 이름
input_filename = "/home/robros/model_uncertainty/script/dataset_gathering/csv/joint_data.csv"
output_filename = "/home/robros/model_uncertainty/script/dataset_gathering/csv/test.csv"
 
# 가속도 계산 및 파일 쓰기
calculate_joint_accelerations(input_filename, output_filename)