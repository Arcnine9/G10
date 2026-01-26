import matplotlib.pyplot as plt
import numpy as np

def read_memory_data(filename):
    """读取显存数据文件"""
    memory_values = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('HookID'):
                # 解析格式: HookID X: Y
                parts = line.split(':')
                if len(parts) == 2:
                    memory = int(parts[1].strip())
                    memory_values.append(memory)
    
    return memory_values

def plot_memory_comparison():
    """绘制显存对比图"""
    # 读取数据
    try:
        memory_orig = read_memory_data('origin_mem_estimate.txt')
        memory_g10 = read_memory_data('G10_mem_estimate.txt')
        
        # 转换为GB单位
        memory_orig_gb = [m / (1024**3) for m in memory_orig]
        memory_g10_gb = [m / (1024**3) for m in memory_g10]
        
    except FileNotFoundError as e:
        print(f"错误: 无法找到文件 - {e}")
        return
    except Exception as e:
        print(f"错误: 读取文件时发生问题 - {e}")
        return
    
    # 设置图形大小和样式
    plt.figure(figsize=(10, 6))
    
    # 创建时间轴（简单的序列）
    time_axis = list(range(len(memory_orig_gb)))
    
    # 绘制两条细线，使用不同的透明度让重合部分可见
    plt.plot(time_axis, memory_orig_gb, 'b-', linewidth=0.8, alpha=0.8, label='Original')
    plt.plot(time_axis, memory_g10_gb, 'r-', linewidth=0.8, alpha=0.8, label='After migration')
    
    # 添加14GB阈值线
    plt.axhline(y=14, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # 设置图表属性
    plt.xlabel('Time', fontsize=24, fontweight='bold')
    plt.ylabel('Memory Usage (GB)', fontsize=24, fontweight='bold')

    plt.yticks(fontsize=15, fontweight='bold')

    # 设置图例
    plt.legend(loc='upper right', framealpha=0.8, prop={'weight': 'bold', 'size': 15})
    
    # 设置网格（更细更淡）
    plt.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    # 设置y轴范围
    plt.ylim(0, max(max(memory_orig_gb), max(memory_g10_gb)) * 1.05)
    
    # 隐藏x轴的具体刻度值
    plt.xticks([])
    
    # 添加一些美化
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('memory_comparison_clean.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 'memory_comparison_clean.png'")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    plot_memory_comparison()