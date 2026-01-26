import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('parsed_tensor_lifecycle.csv')

# 确保 inactive_time 是数值类型
df['inactive_time'] = pd.to_numeric(df['inactive_time'], errors='coerce')

# 移除 inactive_time 中的 NaN 值和 inactive_time == 0 的行
df = df.dropna(subset=['inactive_time'])
df = df[df['inactive_time'] > 0]

# 计算轴的范围
min_inactive_time = df['inactive_time'].min() * 0.9
max_inactive_time = df['inactive_time'].max() * 1.1
min_size = df['size_B'].min() * 0.9
max_size = df['size_B'].max() * 1.1

# 绘制散点图
plt.figure(figsize=(10, 8))
plt.scatter(df['inactive_time'], df['size_B'], s=10, color='blue', alpha=0.7)

# 设置对数坐标轴
plt.xscale('log')
plt.yscale('log')

# 设置标题和坐标轴标签
plt.title('Distribution of Inactive Periods of Tensors for InceptionV3', fontsize=26, fontweight='bold')
plt.xlabel('Inactive Time (µs)', fontsize=24, fontweight='bold')
plt.ylabel('Size (byte)', fontsize=24, fontweight='bold')

plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')

# 设置网格
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 设置坐标轴范围
plt.xlim(min_inactive_time, max_inactive_time)
plt.ylim(min_size, max_size)

# 保存图像
plt.savefig('Inception_tensor_size_vs_inactive_time.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()