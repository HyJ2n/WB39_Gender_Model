import os
import re

def adjust_number_in_line(line):
    # 정규식을 사용하여 줄의 맨 앞에 있는 숫자를 찾습니다.
    match = re.match(r'(\d+)', line)
    if match:
        # 숫자를 -2 합니다.
        number = int(match.group(1)) - 80
        # 수정된 숫자와 원래 줄의 나머지를 결합합니다.
        return re.sub(r'^\d+', str(number), line, 1)
    else:
        # 숫자가 없는 줄은 그대로 반환합니다.
        return line

def process_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    with open(filepath, 'w') as file:
        for line in lines:
            adjusted_line = adjust_number_in_line(line)
            file.write(adjusted_line)

def process_all_txt_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            process_file(filepath)

# 사용자가 원하는 폴더 경로를 지정합니다.
folder_path = r'C:\Users\admin\Desktop\Project\Main\Gender\gender\valid\labels'  # 여기 폴더 경로를 입력하세요
process_all_txt_files_in_folder(folder_path)