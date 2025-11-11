import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

# 读取数据
df = pd.read_csv('parsed_tensor_lifecycle.csv')

# 确保 inactive_time 是数值类型
df['inactive_time'] = pd.to_numeric(df['inactive_time'], errors='coerce')

# 移除 inactive_time 中的 NaN 值和 inactive_time == 0 的行
df = df.dropna(subset=['inactive_time'])
df = df[df['inactive_time'] > 0]

# 生成 CDF
inactive_times = df['inactive_time'].sort_values()
cdf = np.arange(1, len(inactive_times)+1) / len(inactive_times) * 100  # 转换为百分比

# 画图
plt.figure(figsize=(8, 6))
plt.plot(cdf, inactive_times, lw=2, marker='o', color='blue')  # 使用圆点标记每个数据点
plt.xscale('linear')  # 使用线性横轴
plt.yscale('log')  # 纵轴为对数
plt.xlabel('CDF (%)')
plt.ylabel('Tensor Inactive Period Total (µs)')
plt.title('Distribution of Tensor Inactive Period Total Lengths (Excluding Zero Inactive Time)')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.savefig('tensor_inactive_periods_cdf_exclude_zero.png', dpi=300, bbox_inches='tight')  # 保存图像
plt.show()