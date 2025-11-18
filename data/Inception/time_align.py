#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv

KERNEL_LIST_FILE = 'kernels_name.config'
KERNEL_TIME_FILE = 'output.csv'
OUTPUT_FILE = 'align_time.txt'
WINDOW_SIZE = 4  # 每个ReluGrad底层窗口大小

def parse_kernel_list(file_path):
    lst = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Kernel ID:"):
                parts = line.split(',')
                kid = int(parts[0].split(':')[1].strip())
                name = parts[1].split(':')[1].strip()
                lst.append((kid, name))
    return lst

def parse_kernel_time(file_path):
    kt = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            kid_str, name, t_str = row
            try:
                kid = int(kid_str)
                t = float(t_str)
            except:
                continue
            kt.append([kid, name, t])
    return kt

def align_kernel_time(kernel_list, kernel_time):
    i = 0  # pointer for kernel_time
    j = 0  # pointer for kernel_list

    while i < len(kernel_time) and j < len(kernel_list):
        list_kid, list_name = kernel_list[j]

        if list_name != 'A_ReluGrad':
            j += 1
            continue

        # 找到 kernel_time 中对应的 ReluGrad
        while i < len(kernel_time) and kernel_time[i][1] != 'ReluGrad':
            i += 1
        if i >= len(kernel_time):
            break

        # 窗口前3行累加 kernel list中没有的 Slice/Add, 需要对kernel_list中存在的slice/add进行列表
        start_win = max(0, i - 5)
        slice_add_set = {name for _, name in kernel_list[max(0, j-5):j+1] if name in ("A_Concat_Backward", "A_Add")}
        mapping = {"A_Add": "Add", "A_Concat_Backward": "Slice"}
        mapped_set = {mapping[name] for name in slice_add_set if name in mapping}
        extra_time = 0.0
        to_delete = []
        for idx in range(start_win, i):
            if kernel_time[idx][1] in ('Slice', 'Add'):
                if kernel_time[idx][1] not in mapped_set:
                    extra_time += kernel_time[idx][2]
                    to_delete.append(idx)
                mapped_set.discard(kernel_time[idx][1])

        # 删除多余行，从后往前
        for idx in reversed(to_delete):
            del kernel_time[idx]
            i -= 1

        # 累加到当前 ReluGrad
        kernel_time[i][2] += extra_time

        i += 1
        j += 1

    return kernel_time

def main():
    kernel_list = parse_kernel_list(KERNEL_LIST_FILE)
    kernel_time = parse_kernel_time(KERNEL_TIME_FILE)

    aligned_kernel_time = align_kernel_time(kernel_list, kernel_time)
    for new_id, row in enumerate(aligned_kernel_time):
        row[0] = new_id  # 假设 kernel_time 是 [[id, name, time], ...] 的列表

    # 输出完整 kernel_time（已删除冗余 Slice/Add）
    # with open(OUTPUT_FILE, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     for row in aligned_kernel_time:
    #         writer.writerow(row)

    with open(OUTPUT_FILE, 'w') as f:
        for i, kt in enumerate(aligned_kernel_time):
            f.write(f"{i:03d} {kt[2]:.10f} ms\n")


    print(f'Aligned kernel_time written to {OUTPUT_FILE}, total rows: {len(aligned_kernel_time)}')

if __name__ == '__main__':
    main()