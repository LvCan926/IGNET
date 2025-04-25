import pandas as pd

# 正确读取数据（跳过首行列名）
df = pd.read_csv('Data/code/BJ_MGF_processed_384.txt', 
                 sep='\t',
                 header=None,  # 文件没有列名
                 names=['entity_id', 'track_id', 'time', 'lon', 'lat'],  # 手动指定列名
                 skiprows=1,  # 跳过首行（包含列名'time'）
                 dtype={'time': int})

# 验证数据
print("数据类型检查:\n", df.dtypes)
print("\nTime列示例数据:\n", df['time'].head(3))

# 核心逻辑保持不变
test_points = df[df['time'] < 336].groupby(['entity_id', 'track_id']).apply(
    lambda x: x.nlargest(1, 'time')
).reset_index(drop=True)

mask = df.index.isin(test_points.index)
remaining_df = df[~mask]

train_df = remaining_df[remaining_df['time'] <= 287]
val_df = remaining_df[(remaining_df['time'] >= 288) & (remaining_df['time'] <= 335)]

# 保存结果
train_df.to_csv('Data/code/train.txt', sep='\t', header=False, index=False)
val_df.to_csv('Data/code/val.txt', sep='\t', header=False, index=False)
test_points.to_csv('Data/code/test.txt', sep='\t', header=False, index=False)

# 数据验证
print("\n训练集时间范围:", train_df['time'].min(), "~", train_df['time'].max())
print("验证集时间范围:", val_df['time'].min(), "~", val_df['time'].max())
print("测试集时间范围:", test_points['time'].min(), "~", test_points['time'].max())
print("\n测试集样例:")
print(test_points[['entity_id', 'track_id', 'time']].head(3))
