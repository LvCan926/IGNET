def find_min_max_lon_lat(file_path):
    # 初始化变量
    min_lon = None
    max_lon = None
    min_lat = None
    max_lat = None

    with open(file_path, 'r') as f:
        # 跳过标题行
        next(f)
        for line in f:
            # 分割列
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue  # 跳过不完整的行
            lon_str = parts[3]
            lat_str = parts[4]
            try:
                lon = float(lon_str)
                lat = float(lat_str)
            except ValueError:
                # 处理无法转换为浮点数的情况
                print(f"无法转换数值的行：{line}")
                continue

            # 更新经度的最小和最大值
            if min_lon is None or lon < min_lon:
                min_lon = lon
            if max_lon is None or lon > max_lon:
                max_lon = lon

            # 更新纬度的最小和最大值
            if min_lat is None or lat < min_lat:
                min_lat = lat
            if max_lat is None or lat > max_lat:
                max_lat = lat

    return min_lon, max_lon, min_lat, max_lat

# 文件路径
file_path = 'Data/BJ_MGF_processed_2880.txt'

# 获取结果
min_lon, max_lon, min_lat, max_lat = find_min_max_lon_lat(file_path)

# 打印结果
print(f"经度最小值：{min_lon}")
print(f"经度最大值：{max_lon}")
print(f"纬度最小值：{min_lat}")
print(f"纬度最大值：{max_lat}")
