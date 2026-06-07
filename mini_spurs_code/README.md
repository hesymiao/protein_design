# run_pplu_2G6X_chainB_scoring

基于 SPURS (Nature Communications 2025) 对 ppluGFP 变体进行 ΔΔG 热稳定性评估。

## 目录结构

```
protein_design_toolkit/
├── README.md
├── requirements.txt
├── spurs/                               # SPURS 模型代码 (Python 包)# 需要在官网上下载
├── scripts/
│   └── run_pplu_2G6X_chainB_scoring.py  # ★ 唯一脚本
├── data/                                # 放输入文件
└── output/                              # 结果输出
```

## 环境配置

```bash
conda create -n spurs python=3.7 -y
conda activate spurs
pip install -r requirements.txt
```

## 运行

### 1. 确认路径

编辑脚本顶部，确认 `exp_dir` 指向你的 checkpoint：

```python
# Windows:
exp_dir = 'checkpoints/spurs'
# Linux 服务器:
# exp_dir = 'checkpoints/spurs'
```

### 2. 放输入文件到 `data/` 目录

- `pplu_generate_most3.csv`
- `pplu_generate.csv`
- `GFP_data.xlsx`

### 3. 运行

```bash
cd scripts
python run_pplu_2G6X_chainB_scoring.py
```

首次运行会自动下载 2G6X.pdb 和 ESM2-650M 模型 (~2.4 GB)。

## 输出

- 各 CSV 原地新增列：`ddg_vs_ppluWT`, `n_mut_vs_ppluWT`, `n_skipped_no_struct`
- `GFP_data_pplu_scored.csv` — xlsx ppluGFP 部分
- `ppluGFP_ddg_matrix.npy` — L×20 ΔΔG 矩阵

## 依赖的已有文件

| 文件 | 大小 | 位置 |
|------|------|------|
| best.ckpt | 1.9 GB | 放入 `checkpoints/spurs/checkpoints/best.ckpt` |
| ESM2-650M | 2.6 GB | 放入 `checkpoints/esm2_t33_650M_UR50D.pt` 即可，脚本自动拷贝到缓存 |

spurs模型下载方式：
# Tested on Ubuntu 20.04, the setup completes within minutes.
conda create -n spurs python=3.7 pip
conda activate spurs

pip install -e .

pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install git+https://github.com/facebookresearch/esm.git

wget https://www.dropbox.com/scl/fi/uo4e6lvptyy9df5xfulsc/data.tar.gz?rlkey=voi6fxu6ojbzwdk67jlooy8kb&st=4iinnpbc&dl=0
tar -xzvf data.tar.gz


