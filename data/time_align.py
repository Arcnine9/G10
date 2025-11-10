#!/usr/bin/env python3
"""
ResNet Bottleneck 自动对齐脚本（无需手动标注空行）
核心：自动识别Bottleneck边界，整组重排
"""

import sys
from typing import List, Tuple, Optional

class BottleneckInfo:
    """自动识别的Bottleneck元数据"""
    def __init__(self, g10_start: int, trace_start: int, size: int):
        self.g10_start = g10_start      # 在G10序列中的起始位置
        self.trace_start = trace_start  # 在trace中的起始位置
        self.size = size                # G10中该Bottleneck包含的算子数

def parse_file(file_path: str, is_g10: bool) -> Tuple[List[str], List[str]]:
    """
    统一解析函数：
    - 对于G10文件：返回 (算子类型列表, 原始行列表)
    - 对于trace文件：返回 (算子类型列表, 原始行列表)
    """
    types, lines = [], []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        if not is_g10:
            header = next(f).strip()
            lines.append(header)
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if is_g10:
                # G10格式: "Kernel ID: 0, Name: A_Conv2D"
                if 'Name:' not in line:
                    continue
                name = line.split('Name:')[1].strip()
                if 'Conv2D' in name:
                    types.append('CONV')
                elif 'BNTraining_Forward' in name:
                    types.append('BN')
                elif 'Relu' in name and 'Grad' not in name:
                    types.append('RELU')
                elif 'Add' in name:
                    types.append('ADD')
                elif 'MaxPool' in name:
                    types.append('POOL')
                lines.append(line)
            else:
                # Trace格式: "0,Conv_Forward,136.422,Conv_Forward"
                parts = line.split(',')
                if len(parts) < 3:
                    continue
                
                op_type = parts[1].strip()
                type_map = {
                    'Conv_Forward': 'CONV',
                    'BNTraining_Forward': 'BN',
                    'Relu': 'RELU',
                    'Add': 'ADD',
                    'MaxPool_Forward': 'POOL'
                }
                
                if op_type in type_map:
                    types.append(type_map[op_type])
                    lines.append(line)
    
    return types, lines

def find_next_add(g10_types: List[str], start: int) -> Optional[int]:
    """从start位置开始找下一个ADD的位置"""
    for i in range(start + 6, min(start + 20, len(g10_types))):
        if g10_types[i] == 'ADD':
            return i
    return None

def auto_detect_bottlenecks(g10_types: List[str], trace_types: List[str]) -> List[BottleneckInfo]:
    """
    自动检测Bottleneck边界：
    1. 在G10中查找 CONV-BN-CONV-BN-CONV-BN 模式
    2. 在trace中同步定位对应位置
    3. 返回Bottleneck元数据列表
    """
    bottlenecks = []
    g10_ptr, trace_ptr = 0, 0
    
    print("\n🤖 自动检测Bottleneck边界...")
    
    while g10_ptr < len(g10_types) - 5:
        # Bottleneck开始的模式识别
        if (g10_types[g10_ptr:g10_ptr+6] == ['CONV', 'BN', 'CONV', 'BN', 'CONV', 'BN']):
            # 找下一个ADD确定Bottleneck结束
            add_pos = find_next_add(g10_types, g10_ptr)
            if not add_pos:
                g10_ptr += 1
                continue
            
            # 判断是否有downsample: ADD前是否有额外CONV
            has_downsample = 'CONV' in g10_types[g10_ptr+6:add_pos]
            
            # 计算Bottleneck在G10中的长度
            # 主分支6个 + (downsample 2个) + RELU + ADD
            size = 8 + (2 if has_downsample else 0)
            
            # 在trace中定位起始位置（假设与G10大致对齐）
            # trace中Bottleneck特征是密集的CONV-BN对
            while trace_ptr < len(trace_types) - 6:
                if (trace_types[trace_ptr:trace_ptr+6].count('CONV') >= 3 and
                    trace_types[trace_ptr:trace_ptr+6].count('BN') >= 3):
                    break
                trace_ptr += 1
            
            bottleneck = BottleneckInfo(g10_ptr, trace_ptr, size)
            bottlenecks.append(bottleneck)
            
            print(f"  🔍 Bottleneck #{len(bottlenecks)}: "
                  f"G10起点={g10_ptr}, Trace起点={trace_ptr}, "
                  f"Downsample={has_downsample}, 长度={size}")
            
            # 跳过当前Bottleneck，继续检测下一个
            g10_ptr = add_pos + 1  # 跳到ADD之后
            trace_ptr += 10  # 粗略跳过当前Bottleneck区域
        else:
            g10_ptr += 1
    
    return bottlenecks

def align_bottleneck_group(bottleneck: BottleneckInfo,
                          g10_lines: List[str],
                          trace_ops: List[Tuple[int, str, str, str]],
                          start_trace_idx: int) -> Tuple[List[str], int]:
    """
    对齐单个Bottleneck组：
    1. 收集该组在trace中的所有算子
    2. 按G10期望的顺序重排
    3. 返回(对齐后的行, 新的trace_idx)
    """
    print(f"\n{'='*60}")
    print(f"🔗 处理Bottleneck组")
    print(f"   G10起点: {bottleneck.g10_start}")
    print(f"   Trace起点: {start_trace_idx}")
    print(f"   期望大小: {bottleneck.size}")
    print(f"{'='*60}")
    
    # 收集该组的所有trace算子
    collected = []
    trace_idx = start_trace_idx
    
    # Bottleneck在trace中的结束标志：遇到ADD之后的RELU
    found_add = False
    
    while trace_idx < len(trace_ops):
        op_id, op_type, raw_line, _ = trace_ops[trace_idx]
        collected.append(trace_ops[trace_idx])
        trace_idx += 1
        
        print(f"  收集: ID={op_id}, 类型={op_type}")
        
        if op_type == 'ADD':
            found_add = True
        
        # 结束条件：找到ADD后的RELU，或收集到足够算子
        if found_add and op_type == 'RELU':
            break
        
        if len(collected) >= 12:  # 安全边界
            break
    
    # 按G10期望顺序重排
    # G10顺序: [主分支6个] + [downsample 2个] + [RELU] + [ADD]
    result = []
    used = set()
    
    # 1. 主分支3个CONV-BN对
    for i in range(3):
        conv_idx = -1
        for idx, (_, op_type, _, _) in enumerate(collected):
            if idx not in used and op_type == 'CONV':
                conv_idx = idx
                break
        
        if conv_idx == -1:
            raise RuntimeError(f"找不到主分支第{i+1}个CONV")
        
        # 找对应的BN
        bn_idx = conv_idx + 1
        while bn_idx < len(collected) and collected[bn_idx][1] != 'BN':
            bn_idx += 1
        
        result.append(collected[conv_idx][2])
        result.append(collected[bn_idx][2])
        used.update([conv_idx, bn_idx])
        print(f"  ✅ 主分支{i+1}: CONV(ID={collected[conv_idx][0]}) -> BN(ID={collected[bn_idx][0]})")
    
    # 2. Downsample分支（如果有）
    if bottleneck.size > 8:  # 有downsample
        ds_conv_idx = -1
        for idx, (_, op_type, _, _) in enumerate(collected):
            if idx not in used and op_type == 'CONV':
                ds_conv_idx = idx
                break
        
        if ds_conv_idx != -1:
            ds_bn_idx = ds_conv_idx + 1
            while ds_bn_idx < len(collected) and collected[ds_bn_idx][1] != 'BN':
                ds_bn_idx += 1
            
            result.append(collected[ds_conv_idx][2])
            result.append(collected[ds_bn_idx][2])
            used.update([ds_conv_idx, ds_bn_idx])
            print(f"  ✅ Downsample: CONV(ID={collected[ds_conv_idx][0]}) -> BN(ID={collected[ds_bn_idx][0]})")
    
    # 3. RELU（G10中在ADD之前）
    relu_idx = -1
    for idx, (_, op_type, _, _) in enumerate(collected):
        if idx not in used and op_type == 'RELU':
            relu_idx = idx
            break
    
    if relu_idx != -1:
        result.append(collected[relu_idx][2])
        used.add(relu_idx)
        print(f"  ✅ 最终RELU: ID={collected[relu_idx][0]}")
    
    # 4. ADD
    add_idx = -1
    for idx, (_, op_type, _, _) in enumerate(collected):
        if idx not in used and op_type == 'ADD':
            add_idx = idx
            break
    
    if add_idx != -1:
        result.append(collected[add_idx][2])
        used.add(add_idx)
        print(f"  ✅ ADD: ID={collected[add_idx][0]}")
    
    # 跳过的中间RELU
    for idx, (_, op_type, _, _) in enumerate(collected):
        if idx not in used and op_type == 'RELU':
            print(f"  ❌ 跳过中间RELU: ID={collected[idx][0]}")
            used.add(idx)
    
    return result, trace_idx

def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python align_auto_bottleneck.py <g10_file> <time_file> [output_file]")
        print("示例: python align_auto_bottleneck.py kernel_name_g10.txt kernel_time.txt output.txt")
        sys.exit(1)
    
    g10_file = sys.argv[1]
    time_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else 'kernel_time_aligned.txt'
    
    print("="*70)
    print("ResNet Bottleneck 自动对齐工具 (无需手动标注)")
    print("="*70)
    
    try:
        # 1. 解析文件
        print("\n📦 解析G10结构...")
        g10_types, g10_lines = parse_file(g10_file, is_g10=True)
        print(f"   ✅ G10算子数: {len(g10_types)}")
        
        print("\n📊 解析Trace...")
        trace_types, trace_lines = parse_file(time_file, is_g10=False)
        header = trace_lines[0]
        trace_ops = []
        for i in range(1, len(trace_types)):
            # 格式: (原始行号, 类型, raw_line, 用于调试的额外信息)
            trace_ops.append((i, trace_types[i-1], trace_lines[i], ""))
        
        print(f"   ✅ Trace算子数: {len(trace_ops)}")
        
        # 2. 自动检测Bottleneck
        bottlenecks = auto_detect_bottlenecks(g10_types, trace_types)
        print(f"\n🎯 共识别 {len(bottlenecks)} 个Bottleneck结构")
        
        # 3. 对齐处理
        print("\n🔄 开始对齐...")
        aligned = [header]
        current_trace_idx = 0
        
        for i, bottleneck in enumerate(bottlenecks):
            # 处理Bottleneck前的非Bottleneck算子
            while current_trace_idx < bottleneck.trace_start:
                if trace_ops[current_trace_idx][1] != 'RELU':  # 跳过非Bottleneck的多余RELU
                    aligned.append(trace_ops[current_trace_idx][2])
                current_trace_idx += 1
            
            # 处理当前Bottleneck组
            group_lines, new_trace_idx = align_bottleneck_group(
                bottleneck, g10_lines, trace_ops, current_trace_idx)
            aligned.extend(group_lines)
            current_trace_idx = new_trace_idx
        
        # 处理剩余算子
        while current_trace_idx < len(trace_ops):
            if trace_ops[current_trace_idx][1] != 'RELU':
                aligned.append(trace_ops[current_trace_idx][2])
            current_trace_idx += 1
        
        # 4. 写入结果
        with open(output_file, 'w') as f:
            f.write('\n'.join(aligned) + '\n')
        
        # 5. 统计
        print("\n" + "="*70)
        print(f"✅ 对齐完成: {output_file}")
        print(f"   G10算子数: {len(g10_types)}")
        print(f"   对齐后算子数: {len(aligned)-1}")
        print(f"   匹配状态: {'✓ 完美对齐' if len(aligned)-1 == len(g10_types) else '⚠️  数量不符'}")
        print("="*70)
        
    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()