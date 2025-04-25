# 增强版OD矩阵计算与可视化

这个项目提供了两个主要脚本：
1. `enhanced_od_matrix.py` - 用于计算增强版OD矩阵
2. `visualize_od_matrix.py` - 用于可视化OD矩阵结果

## 增强版OD矩阵

增强版OD矩阵计算每个时间步（减去偏移量1020映射到0~19），每个网格车辆数到每个网格车辆数的占比，以及所花的时间步。例如，对于时间步1045，计算的是从该时间步出发的车辆到达其他网格的占比及所需的平均时间。

**特性**：
- 该实现会统计出发和到达相同网格的情况，以处理那些速度较慢、需要经过较长时间才能到达其他区域的车辆
- 轨迹数据按照entity_id（车辆ID）分组处理，而不是track_id
- 支持数据采样功能，便于快速测试和调试

OD矩阵的格式为：`[时间步, 出发网格, 到达网格, 2]`，其中最后一个维度的两个值分别表示：
- 占比：到达指定网格的车辆数占比
- 平均时间：到达指定网格所需的平均时间步

## 数据格式

输入数据应为带有以下列的制表符分隔文本文件：
```
entity_id    track_id    time    lon    lat
```

例如：
```
3    1    1045    116.4958378    39.90689665
```

## 使用方法

### 1. 计算增强版OD矩阵

```bash
python enhanced_od_matrix.py [参数]
```

可选参数：
- `--input` - 输入轨迹数据文件路径 (默认: train.txt)
- `--output` - 输出OD矩阵文件路径 (默认: enhanced_od_matrix.npy)
- `--time-steps` - 时间步数量 (默认: 20)
- `--time-offset` - 时间步偏移量 (默认: 1020)
- `--grid-size` - 网格大小 (默认: 32)
- `--min-lon` - 最小经度 (默认: 116.0)
- `--max-lon` - 最大经度 (默认: 117.0)
- `--min-lat` - 最小纬度 (默认: 39.6)
- `--max-lat` - 最大纬度 (默认: 40.6)
- `--sample-ratio` - 数据采样比例，用于快速测试 (默认: 1.0，表示使用全部数据)

示例：
```bash
# 使用全部数据
python enhanced_od_matrix.py --input train.txt --output od_matrix.npy --grid-size 64

# 使用10%的数据进行测试
python enhanced_od_matrix.py --input train.txt --output test_od_matrix.npy --sample-ratio 0.1
```

### 2. 可视化OD矩阵

```bash
python visualize_od_matrix.py [参数]
```

可选参数：
- `--input` - 输入OD矩阵文件路径 (默认: enhanced_od_matrix.npy)
- `--output-dir` - 输出可视化目录 (默认: visualization)
- `--time-step` - 要可视化的时间步 (默认: 0)
- `--grid-size` - 网格大小 (默认: 32)

示例：
```bash
python visualize_od_matrix.py --input od_matrix.npy --time-step 5
```

## 可视化结果

脚本将生成以下可视化结果：

1. **时间模式可视化** - 显示不同时间步的活跃OD对数量
2. **平均旅行时间可视化** - 显示不同时间步的平均旅行时间
3. **OD热力图** - 针对特定时间步，生成以下热力图：
   - 目的地热力图：显示到达各目的地的占比之和
   - 出发点热力图：显示从各出发点的占比之和
   - 平均旅行时间热力图：显示到达各目的地的平均旅行时间

## 注意事项

1. 确保安装了所需的Python库：
   ```bash
   pip install numpy pandas matplotlib seaborn
   ```

2. 处理大型数据集时，计算可能需要较长时间。建议先使用小型数据集或设置较小的采样比例进行测试：
   ```bash
   python enhanced_od_matrix.py --sample-ratio 0.01  # 使用1%的数据
   ```

3. 网格大小的选择会影响计算结果和可视化效果，需要根据实际情况进行调整。

4. 经纬度范围的设置需要根据实际数据范围进行调整，以确保有效的网格映射。

5. 通过统计出发和到达相同网格的情况，算法能够捕获在原地停留或缓慢移动的车辆行为模式。

6. 调试技巧：先使用较小的采样比例（如0.01）测试代码是否正常工作，然后再使用全部数据进行计算。 