"""
用 SPURS 测 ppluGFP 所有变体的 ΔΔG
1. pplu_generate_most3.csv / pplu_generate.csv → 已有完整序列
2. GFP_data.xlsx brightness sheet (ppluGFP 部分) → 从突变描述重建序列
"""
import torch
import numpy as np
import pandas as pd
import os
import re
import shutil
from spurs.inference import get_SPURS, parse_pdb

# ============================================================
# 0. 自动处理 ESM2 模型：优先从本地 checkpoints 复制到 torch hub 缓存
# ============================================================
def _ensure_esm2_model(script_dir):
    """确保 ESM2-650M 模型在 torch hub 缓存中"""
    esm2_name = 'esm2_t33_650M_UR50D.pt'
    hub_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
    hub_path = os.path.join(hub_dir, esm2_name)

    if os.path.exists(hub_path):
        return  # 已有

    # 从 toolkit 的 checkpoints/ 找
    toolkit_dir = os.path.dirname(os.path.dirname(os.path.abspath(script_dir)))
    local_path = os.path.join(toolkit_dir, 'checkpoints', esm2_name)
    if os.path.exists(local_path):
        os.makedirs(hub_dir, exist_ok=True)
        print(f"复制 ESM2 模型: {local_path} → {hub_path}")
        shutil.copy2(local_path, hub_path)
    else:
        print(f"ESM2 模型未找到，将由 fair-esm 自动下载")
        print(f"(你也可以手动放入: {local_path})")

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

# ppluGFP WT (from AAseqs)
pplu_WT = "MPAMKIECRITGTLNGVEFELVGGGEGTPEQGRMTNKMKSTKGALTFSPYLLSHVMGYGFYHFGTYPSGYENPFLHAINNGGYTNTRIEKYEDGGVLHVSFSYRYEAGRVIGDFKVVGTGFPEDSVIFTDKIIRSNATVEHLHPMGDNVLVGSFARTFSLRDGGYYSFVVDSHMHFKSAIHPSILQNGGPMFAFRRVEELHSNTELGIVEYQHAFKTPIAFA"

# PDB 只覆盖 pplu_WT 的位置 3-218 (0-indexed: 2..217)
PDB_OFFSET = 2

# ============================================================
# 1. 下载结构 & 跑 SPURS
# ============================================================
def get_pplu_structure():
    pdb_path = '2G6X.pdb'
    if not os.path.exists(pdb_path):
        import urllib.request
        url = 'https://files.rcsb.org/download/2G6X.pdb'
        print(f"下载 {url} ...")
        urllib.request.urlretrieve(url, pdb_path)
        print(f"已保存: {pdb_path}")
    return pdb_path

def run_spurs_pplu(pdb_path):
    # Checkpoint 在 toolkit 自带的 checkpoints 目录
    import sys
    toolkit_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = os.path.join(toolkit_dir, 'checkpoints', 'spurs')
    print(f"experiment dir: {exp_dir}")
    model, cfg = get_SPURS(exp_dir)

    print(f"Parsing {pdb_path}...")
    pdb = parse_pdb(pdb_path, 'ppluGFP', 'A', cfg)

    # 修复 PDB 中未解析残基（gap '-'）
    seq = pdb['seq']
    seq = ''.join(pplu_WT[i] if s == '-' else s for i, s in enumerate(seq))
    pdb['seq'] = seq

    print("预测 ΔΔG 矩阵 (L × 20)...")
    model.eval()
    with torch.no_grad():
        ddg = model(pdb, return_logist=True)

    L = ddg.shape[0]
    print(f"ppluGFP: {L} residues, WT seq = {pdb['seq']}")

    np.save('ppluGFP_ddg_matrix.npy', ddg.cpu().numpy())
    print("已保存: ppluGFP_ddg_matrix.npy")
    return ddg.cpu().numpy(), pdb['seq']

# ============================================================
# 2. 评分函数
# ============================================================
def score_vs_wt(seq, ddg_matrix, wt_seq, offset=PDB_OFFSET):
    """返回 (总 ΔΔG, 突变数, 跳过数)"""
    total = 0.0
    n_mut = 0
    n_skipped = 0
    for i, (w, m) in enumerate(zip(wt_seq, seq)):
        if w == m:
            continue
        n_mut += 1
        mat_idx = i - offset
        if mat_idx < 0 or mat_idx >= ddg_matrix.shape[0]:
            n_skipped += 1
            continue
        try:
            total += ddg_matrix[mat_idx, ALPHABET.index(m)]
        except (ValueError, IndexError):
            n_skipped += 1
    return total, n_mut, n_skipped

# 解析 "A109D:N145D:I187V" 风格突变，应用到 WT 上重建序列
MUT_RE = re.compile(r'^([A-Z])(\d+)([A-Z])$')

def apply_mutations_to_wt(mut_str, wt_seq):
    """从突变字符串重建全长序列"""
    seq = list(wt_seq)
    if mut_str == 'WT' or pd.isna(mut_str):
        return wt_seq
    for m in str(mut_str).split(':'):
        m = m.strip()
        match = MUT_RE.match(m)
        if not match:
            continue
        wt_aa, pos_str, mt_aa = match.groups()
        pos = int(pos_str) - 1  # 1-indexed → 0-indexed
        if pos < 0 or pos >= len(seq):
            continue
        # 不校验 wt_aa 一致性，直接应用（数据本身可能有标注差异）
        seq[pos] = mt_aa
    return ''.join(seq)

# ============================================================
# 3. 批量评分 & 输出
# ============================================================
def score_and_save(df, seq_col, ddg_matrix, wt_seq, output_path, src_label):
    """对 DataFrame 中 seq_col 列每条序列打分，写回文件，打印 Top 10 亮度"""
    print(f"\n{'='*60}")
    print(f"[{src_label}] 共 {len(df)} 条序列")

    ddg_vals, n_mut_vals, n_skip_vals = [], [], []
    for idx, row in df.iterrows():
        seq = row[seq_col]
        ddg, n_mut, n_skip = score_vs_wt(seq, ddg_matrix, wt_seq)
        ddg_vals.append(round(ddg, 4))
        n_mut_vals.append(n_mut)
        n_skip_vals.append(n_skip)

    df['ddg_vs_ppluWT'] = ddg_vals
    df['n_mut_vs_ppluWT'] = n_mut_vals
    df['n_skipped_no_struct'] = n_skip_vals

    # 按亮度降序排列
    for c in ['predicted_brightness', 'Brightness', 'brightness']:
        if c in df.columns:
            df = df.sort_values(c, ascending=False, na_position='last')
            break

    df = df.reset_index(drop=True)
    df.to_csv(output_path, index=False)

    # 统计
    ddg_arr = np.array(ddg_vals)
    stabilizing = (ddg_arr < -0.5).sum()
    neutral = ((ddg_arr >= -0.5) & (ddg_arr <= 0.5)).sum()
    destabilizing = (ddg_arr > 0.5).sum()
    print(f"  稳定化 (<-0.5): {stabilizing} | 中性: {neutral} | 不稳定 (>0.5): {destabilizing}")
    print(f"  ΔΔG 范围: [{ddg_arr.min():+.2f}, {ddg_arr.max():+.2f}]")
    print(f"  已保存: {output_path}")

    # 打印 Top 10 亮度
    brightness_col = None
    for c in ['predicted_brightness', 'Brightness', 'brightness']:
        if c in df.columns:
            brightness_col = c
            break

    if brightness_col:
        b = pd.to_numeric(df[brightness_col], errors='coerce')
        top10_idx = b.nlargest(10).index
        print(f"\n  === Top 10 亮度 [{src_label}] ===")
        print(f"  {'rank':<5} {'ddg':>8} {'brightness':>12}  {'前60aa / 突变'}")
        print(f"  {'-'*55}")
        for rank, i in enumerate(top10_idx, 1):
            row = df.loc[i]
            seq_preview = str(row[seq_col])[:60]
            ddg = row['ddg_vs_ppluWT']
            bright = b[i]
            # 如果有突变信息也显示
            mut_info = ''
            for mc in ['mutations_vs_parent', 'aaMutations']:
                if mc in df.columns and pd.notna(row[mc]):
                    mut_info = f' [{str(row[mc])[:50]}]'
                    break
            print(f"  {rank:<5} {ddg:>+8.2f} {bright:>12.4f}  {seq_preview}{mut_info}")
    else:
        print(f"  (无亮度列，跳过 Top 10)")

    return df

# ============================================================
# 4. 主流程
# ============================================================
if __name__ == '__main__':
    _ensure_esm2_model(__file__)

    BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    # --- 跑 SPURS ---
    pdb_path = get_pplu_structure()
    ddg_matrix, spurs_wt = run_spurs_pplu(pdb_path)
    print(f"ddg_matrix shape: {ddg_matrix.shape}")

    # --- 3.1 处理 pplu_generate.csv 系列 ---
    for csv_name in ['pplu_generate_most3.csv', 'pplu_generate.csv']:
        csv_path = os.path.join(BASE, csv_name)
        if not os.path.exists(csv_path):
            print(f"\n跳过: {csv_path} (文件不存在)")
            continue
        df = pd.read_csv(csv_path)
        score_and_save(df, 'sequence', ddg_matrix, pplu_WT,
                       csv_path, csv_name.replace('.csv', ''))

    # --- 3.2 处理 GFP_data.xlsx (ppluGFP 部分) ---
    xlsx_paths = [
        os.path.join(BASE, 'GFP_data.xlsx'),
        '/data1/whm/synbio competition/GFP_data.xlsx',
    ]
    xlsx_path = None
    for p in xlsx_paths:
        if os.path.exists(p):
            xlsx_path = p
            break

    if xlsx_path:
        print(f"\n{'='*60}")
        print(f"处理: GFP_data.xlsx (brightness sheet, ppluGFP)")
        df_xls = pd.read_excel(xlsx_path, sheet_name='brightness')
        df_pplu = df_xls[df_xls['meiyou'] == 'ppluGFP'].copy()
        print(f"  筛选出 {len(df_pplu)} 条 ppluGFP 数据")

        # 从突变重建序列
        print("  重建序列中...")
        df_pplu['sequence'] = df_pplu['aaMutations'].apply(
            lambda m: apply_mutations_to_wt(m, pplu_WT)
        )

        out_path = os.path.join(BASE, 'GFP_data_pplu_scored.csv')
        score_and_save(df_pplu, 'sequence', ddg_matrix, pplu_WT,
                       out_path, 'GFP_data.xlsx::ppluGFP')
    else:
        print(f"\nGFP_data.xlsx 未找到，跳过")

    print(f"\n{'='*60}")
    print("全部完成！")
