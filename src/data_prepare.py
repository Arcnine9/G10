#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd

IN_CSV  = '../data/kernel_details.csv'
OUT_CSV = '../data/output.csv'
SEP     = ','                 # 如果是 tab 改成 '\t'

# 需要累加的列
SUM_COLS = ['Duration(us)']

# ---------- 工具 ----------
def replace_rows(df, idx, span, new_rows):
    """删掉 df[idx:idx+span]，插入 new_rows（list[Series]）"""
    return pd.concat([df.iloc[:idx],
                      pd.DataFrame(new_rows),
                      df.iloc[idx+span:]], ignore_index=True)

# ---------- Forward 合并 ----------
def fold_conv_forward(df):
    idx = 0
    while idx <= len(df) - 5:
        t = df.iloc[idx:idx+5]['Type'].tolist()
        if t == ['TransData','TransData','Conv2D','TransData','Add']:
            summed = df.iloc[idx:idx+5][SUM_COLS].sum()
            row = df.iloc[idx].copy()
            row[SUM_COLS] = summed
            # row['Name'] = row['Type'] = 'Conv_Forward'
            df = replace_rows(df, idx, 5, [row])
            idx += 1
            continue
        idx += 1
    return df

def fold_bn_forward(df):
    idx = 0
    while idx <= len(df) - 3:
        t = df.iloc[idx:idx+3]['Type'].tolist()
        if t == ['MemSet','BNTrainingReduce','BNTrainingUpdate']:
            summed = df.iloc[idx:idx+3][SUM_COLS].sum()
            row = df.iloc[idx].copy()
            row[SUM_COLS] = summed
            # row['Name'] = row['Type'] = 'BNTraining_Forward'
            df = replace_rows(df, idx, 3, [row])
            idx += 1
            continue
        idx += 1
    return df

def fold_maxpool_forward(df):
    """
    1 TransData + MaxPoolWithArgmaxV1 + 1 Post TransData
    """
    idx = 0
    while idx <= len(df) - 3:
        t = df.iloc[idx:idx+3]['Type'].tolist()
        if t == ['TransData','MaxPoolWithArgmaxV1','TransData']:
            summed = df.iloc[idx:idx+3][SUM_COLS].sum()
            row = df.iloc[idx].copy()
            row[SUM_COLS] = summed
            # row['Name'] = row['Type'] = 'MaxPool_Forward'
            df = replace_rows(df, idx, 3, [row])
            idx += 1
            continue
        idx += 1
    return df
# ---------- makeloss 合并 ----------
def fold_make_loss(df):
    """
    聚合 MakeLoss 区间：
    从第一个 LogSoftmaxV2 开始，直到 LogSoftmaxGrad 或 NLLLossGrad 结束。
    期间可包含辅助算子（Fill / OnesLike / Cast / MemSet 等）。
    """
    make_loss_ops = [
        'LogSoftmaxV2', 'OnesLike', 'MemSet', 'NLLLoss',
        'Fill', 'Cast', 'NLLLossGrad', 'LogSoftmaxGrad'
    ]
    start_op = 'LogSoftmaxV2'
    end_ops = {'LogSoftmaxGrad'}

    idx = 0
    while idx < len(df):
        # 检测 MakeLoss 起点
        if df.iloc[idx]['Type'] == start_op:
            start = idx
            end = idx

            # 向后查找，直到找到结束算子
            while end + 1 < len(df):
                op_type = df.iloc[end + 1]['Type']
                end += 1
                if op_type in end_ops:
                    # 继续到最后一个结束算子
                    while end + 1 < len(df) and df.iloc[end + 1]['Type'] in end_ops:
                        end += 1
                    break
                # 如果出现非 make_loss 算子且没到结束点，退出
                if op_type not in make_loss_ops:
                    print("出现非 make_loss 算子且没到结束点")
                    break

            # 聚合区间 [start, end]
            summed = df.iloc[start:end + 1][SUM_COLS].sum()
            row = df.iloc[start].copy()
            row[SUM_COLS] = summed
            # row['Name'] = row['Type'] = 'MakeLoss'
            df = replace_rows(df, start, end - start + 1, [row])

            idx = start + 1
            continue

        idx += 1

    return df


# ---------- Backward 合并 ----------
def fold_conv_backward(df):
    idx = 0
    while idx <= len(df) - 10:
        t = df.iloc[idx:idx+10]['Type'].tolist()
        if (t[:2] == ['TransData','TransData'] and
            t[2] == 'Conv2DBackpropInput' and
            t[3:6] == ['TransData','TransData','TransData'] and
            t[6] == 'MemSet' and
            t[7] == 'Conv2DBackpropFilter' and
            t[8] == 'TransData' and
            t[9] == 'TensorMove'):
            sum_in  = df.iloc[idx:idx+5][SUM_COLS].sum()
            sum_flt = df.iloc[idx+5:idx+10][SUM_COLS].sum()
            row_in  = df.iloc[idx].copy()
            row_flt = df.iloc[idx+7].copy()
            row_in[SUM_COLS]  = sum_in
            row_flt[SUM_COLS] = sum_flt
            # row_in['Name']  = row_in['Type']  = 'Conv2DBackpropInput'
            # row_flt['Name'] = row_flt['Type'] = 'Conv2DBackpropFilter'
            df = replace_rows(df, idx, 10, [row_in, row_flt])
            idx += 2
            continue
        idx += 1
    return df

def fold_bn_backward(df):
    idx = 0
    while idx <= len(df) - 3:
        t = df.iloc[idx:idx+3]['Type'].tolist()
        if t == ['MemSet','BNTrainingUpdateGrad','BNTrainingReduceGrad']:
            summed = df.iloc[idx:idx+3][SUM_COLS].sum()
            row = df.iloc[idx].copy()
            row[SUM_COLS] = summed
            # row['Name'] = row['Type'] = 'BNTrainingGrad'
            df = replace_rows(df, idx, 3, [row])
            idx += 1
            continue
        idx += 1
    return df

def fold_maxpool_backward(df):
    idx = 0
    while idx <= len(df) - 3:
        t = df.iloc[idx:idx+3]['Type'].tolist()
        if t == ['TransData','TransData','MaxPoolGradWithArgmaxV1']:
            summed = df.iloc[idx:idx+3][SUM_COLS].sum()
            row = df.iloc[idx].copy()
            row[SUM_COLS] = summed
            # row['Name'] = row['Type'] = 'MaxPoolGradWithArgmaxV1'
            df = replace_rows(df, idx, 3, [row])
            idx += 1
            continue
        idx += 1
    return df

def fold_linear_backward(df):
    """
    MemSet + MatMulV2(dW) + MemSet + MatMulV2(dx) + ReduceSum(db)
    拆成 3 条新记录：
      MatMulV2_dW / MatMulV2_dx / ReduceSum_db
    各自 duration 累加对应行。
    """
    idx = 0
    while idx <= len(df) - 5:
        t = df.iloc[idx:idx+5]['Type'].tolist()
        if (t == ['MemSet','MatMulV2','MemSet','MatMulV2','ReduceSum']):
            # 第 1 个 MatMulV2（dW）
            sum_dw = df.iloc[[idx, idx+1]][SUM_COLS].sum()
            row_dw = df.iloc[idx+1].copy()
            row_dw[SUM_COLS] = sum_dw
            # row_dw['Name'] = row_dw['Type'] = 'MatMulV2_dW'
            # 第 2 个 MatMulV2（dx）
            sum_dx = df.iloc[[idx+2, idx+3]][SUM_COLS].sum()
            row_dx = df.iloc[idx+3].copy()
            row_dx[SUM_COLS] = sum_dx
            # row_dx['Name'] = row_dx['Type'] = 'MatMulV2_dx'
            # ReduceSum（db）
            sum_db = df.iloc[[idx+4]][SUM_COLS].sum()
            row_db = df.iloc[idx+4].copy()
            row_db[SUM_COLS] = sum_db
            # row_db['Name'] = row_db['Type'] = 'ReduceSum_db'
            df = replace_rows(df, idx, 5, [row_dw, row_dx, row_db])
            idx += 3
            continue
        idx += 1
    return df

# ---------- 主流程 ----------
def main():
    df = pd.read_csv(IN_CSV, sep=SEP).sort_values('Start Time(us)').reset_index(drop=True)
    # 这里要删去无关列
    df = df.loc[:,['Type','Duration(us)']]
    # 这里选取一轮的行
    mask = df['Type'] == 'ForeachAddList'
    if mask.any():  # 如果确实存在
        first_idx = mask.idxmax()  # 第一个 True 的索引
        df = df.loc[:first_idx - 1]  # 取前面的所有行
    else:
        print("⚠️ 未找到 Type == 'ForeachAddList'，保留全部数据。")

    df = fold_conv_forward(df)
    df = fold_bn_forward(df)
    df = fold_maxpool_forward(df)

    df = fold_make_loss(df)
    
    df = fold_conv_backward(df)
    df = fold_bn_backward(df)
    df = fold_maxpool_backward(df)
    df = fold_linear_backward(df)
    df = df.loc[:, ['Duration(us)']]
    df['Duration(ms)'] = df['Duration(us)'] / 1000.0  # 转换为毫秒

    # 输出为指定格式
    lines = [
        f"{i:04d} {v:.6f} ms"
        for i, v in enumerate(df['Duration(ms)'])
    ]

    # 写入文本文件（或直接打印）
    with open(OUT_CSV.replace('.csv', '.txt'), 'w') as f:
        f.write('\n'.join(lines))

    print(f'folded txt -> {OUT_CSV.replace(".csv", ".txt")}   rows: {len(df)}')

if __name__ == '__main__':
    main()