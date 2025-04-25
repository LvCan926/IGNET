import os

def process_file(file_path):
    # 读取文件内容
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 处理每一行
    processed_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0:
            # 将第一列除以60并转换为整数
            parts[0] = str(int(float(parts[0]) / 60))
        processed_lines.append(' '.join(parts) + '\n')
    
    # 写回文件
    with open(file_path, 'w') as f:
        f.writelines(processed_lines)

def main():
    # 处理三个文件
    files = [
        'src/data/TP/raw_data/BJTaxi/test/test.txt',
        'src/data/TP/raw_data/BJTaxi/test/train.txt',
        'src/data/TP/raw_data/BJTaxi/test/val.txt'
    ]
    
    for file_path in files:
        if os.path.exists(file_path):
            print(f'处理文件: {file_path}')
            process_file(file_path)
        else:
            print(f'文件不存在: {file_path}')

if __name__ == '__main__':
    main()
