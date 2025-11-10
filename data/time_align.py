#!/usr/bin/env python3
"""
单指针 + 全局有序池（不切组）
用法：
python global_queue_align.py kernel_name.txt kernel_time.txt output.txt
"""
import re, sys
from typing import List, Dict, Tuple
from collections import deque

OpT = Tuple[int, str, float]

# ---------- 解析 ----------
def parse_name(path: str) -> List[str]:
    names = []
    with open(path) as f:
        for line in f:
            m = re.search(r'Name: (A_\w+)', line)
            if m:
                names.append(m.group(1))
    return names

def parse_time(path: str) -> List[OpT]:
    ops = []
    with open(path) as f:
        next(f)
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(',')
            if len(parts) < 3:
                continue
            ops.append((int(parts[0]), parts[1], float(parts[2])))
    return ops

# ---------- 1. 清洗 ----------
def clean_trace(trace: List[OpT]) -> List[OpT]:
    cleaned = []
    first_relu_done = False
    for i, op in enumerate(trace):
        if op[1] == 'Relu':
            if not first_relu_done or (i > 0 and trace[i - 1][1] == 'Add'):
                cleaned.append(op)
                first_relu_done = True
            continue
        if op[1] == 'ReluGrad' and i > 0 and trace[i - 1][1] == 'Conv2DBackpropFilter':
            continue
        cleaned.append(op)
    return cleaned

# ---------- 2. 全局有序池 ----------
def build_queues(cleaned: List[OpT]) -> Dict[str, deque[OpT]]:
    queues: Dict[str, deque[OpT]] = {}
    for op in cleaned:
        if op[1] not in queues:
            queues[op[1]] = deque()
        queues[op[1]].append(op)
    return queues

def align_global_ptr(kn: List[str], queues: Dict[str, deque[OpT]]) -> List[OpT]:
    aligned = []
    for name in kn:
        k_type = map_type(name)
        if not k_type:
            missing = [n for n in kn if not map_type(n)]
            if missing:
                print(f"⚠️  映射缺失：{missing[0]}")
                raise RuntimeError(f"映射表缺失类型 {missing[0]}！请补全。")
        if k_type not in queues or not queues[k_type]:
            raise RuntimeError(f"trace 中找不到类型 {k_type} 的下一行！请检查清洗逻辑或 trace 完整性。")
        aligned.append(queues[k_type].popleft())
    return aligned

# ---------- 映射表（与 trace 原始名字一致） ----------
def map_type(name: str) -> str:
    return {
        # 正向
        'A_Conv2D': 'Conv_Forward',
        'A_BNTraining_Forward': 'BNTraining_Forward',
        'A_Relu': 'Relu',
        'A_Add': 'Add',
        'A_MaxPoolWithArgMaxV1': 'MaxPool_Forward',
        # 过渡
        'A_ReduceMean': 'ReduceMean',
        'A_MatMulV2': 'MatMulV2',
        'A_makeLoss': 'MakeLoss',
        'A_Mul': 'Mul',
        'A_Fill': 'Fill',
        'A_ReduceSum': 'ReduceSum',
        # 反向
        'A_ReluGrad': 'ReluGrad',
        'A_BNTrainingUpdateGrad': 'A_BNTrainingUpdateGrad',
        'A_BNTrainingReduceGrad': 'A_BNTrainingReduceGrad',
        'A_Conv2DBackpropInput': 'Conv2DBackpropInput',
        'A_Conv2DBackpropFilter': 'Conv2DBackpropFilter',
        'A_MaxPoolGradWithArgmaxV1': 'MaxPoolGradWithArgmaxV1',
        'A_Add': 'Add',
    }.get(name, "")


# ---------- 主流程 ----------
def main():
    if len(sys.argv) != 4:
        print("用法: python global_queue_align.py kernel_name.txt kernel_time.txt output.txt")
        sys.exit(1)

    kn_path, kt_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    kernel_names = parse_name(kn_path)
    trace_ops = parse_time(kt_path)

    # 1. 清洗
    cleaned = clean_trace(trace_ops)
    print(f"清洗后 trace 行数: {len(cleaned)}")

    with open("kernel_time_cleaned.txt", 'w') as f:
        f.write("ID,Type,Duration(ms)\n")
        for op in cleaned:
            f.write(f"{op[0]},{op[1]},{op[2]}\n")

    # 2. 全局有序池
    queues = build_queues(cleaned)

    # 3. 单指针对齐
    aligned = align_global_ptr(kernel_names, queues)

    # 4. 输出
    # 4. 输出新格式
    with open(out_path, 'w') as f:
        for idx, op in enumerate(aligned):
            f.write(f"{idx:04d} {op[2]:.6f} ms\n")

    print(f"✅ 全局队列对齐完成 → {out_path}")
    print(f"   kernel_name 算子数: {len(kernel_names)}")
    print(f"   对齐后行数: {len(aligned)}")

if __name__ == "__main__":
    main()