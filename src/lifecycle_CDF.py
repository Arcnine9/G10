import pandas as pd

# 定义列名
columns = ['tensor_id', 'size_B', 'birth_kid', 'death_kid', 'hidden_interval_num', 'inactive_time']

# 初始化数据列表
data = []

# 读取文件
with open('/home/user2/G10/results/Inception/Ascend/tensor_lifecycle.txt', 'r') as file:
    next(file)  # 跳过标题行
    for line in file:
        parts = line.strip().split()
        
        # 检查行是否有足够的字段
        if len(parts) < 5:
            continue  # 跳过格式不正确的行
        
        tensor_id = int(parts[0])
        size_B = int(parts[1])
        birth_kid = int(parts[2])
        death_kid = int(parts[3])
        hidden_interval_num = int(parts[4])
        
        # 计算非激活时间
        inactive_time = 0
        if hidden_interval_num > 0 and len(parts) > 5:
            details = parts[5]
            intervals = details.split(')(')
            for interval in intervals:
                interval = interval.strip('()')
                if ',' in interval:
                    start_kid, end_kid, dur_us = map(float, interval.split(','))
                    inactive_time += dur_us
        
        # 保存数据
        data.append([tensor_id, size_B, birth_kid, death_kid, hidden_interval_num, inactive_time])

# 创建DataFrame
df = pd.DataFrame(data, columns=columns)

# 保存到新的CSV文件以便进一步分析
df.to_csv('Inception_parsed_tensor_lifecycle.csv', index=False)

print("数据已解析并保存到 parsed_tensor_lifecycle.csv")