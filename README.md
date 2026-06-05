# Protein Design Workflow

这份代码的主流程是：

1. 准备亮度预测数据
2. 训练 5-fold 亮度预测模型
3. 用 steer-PLM 生成高亮度候选序列
4. 用 ProteinMPNN 做结构一致性打分
5. 用 Venus-MAXWELL 做进一步打分

## Step 1. 启动环境

先进入对应环境，例如：

```bash
conda activate light_predictor_esm2
```

## Step 2. 准备亮度预测数据

运行：

```bash
bash steer-PLM/light_predictor/avgfp_esm2_lora/run_prepare_pplu_dataset.sh
```

作用：

- 读取原始 GFP 数据
- 构建 `ppluGFP` 的亮度数据表
- 生成训练所需的 `pplu_split.csv`

注意：

- 这一步会自动划分 `train / val / test`
- 这里的 `val` 主要是给训练过程监控使用，实际生成阶段并不直接依赖它

## Step 3. 训练亮度预测模型

运行：

```bash
bash steer-PLM/light_predictor/avgfp_esm2_lora/run_lora_tail_regression_5fold_ensemble.sh
```

作用：

- 基于 `pplu_split.csv` 训练 5 个 fold 的亮度预测模型
- 后续生成阶段会使用这 5 个模型的平均预测结果作为亮度打分

注意：

- 需要把脚本中的数据路径改成你自己的实际路径
- 需要把 ESM2 基座模型路径改成你自己的实际路径

## Step 4. 生成高亮度候选序列

运行：

```bash
bash steer-PLM/Steering-PLMs/generate_5fold.sh
```

作用：

- 基于高亮和低亮序列提取 steering direction
- 从高亮训练样本中提取热点突变位点
- 用 ESM2 在热点位点上提出候选突变
- 用 5-fold 亮度预测器对候选序列打分
- 输出排序后的候选序列表

你可以调的主要参数包括：

- 突变轮次 `rounds`
- 每轮搜索宽度 `beam-width`
- 每轮分支数 `branch-factor`
- 最大突变位点数 `mutation-budget-max`

补充说明：

- 代码会自动标记某条生成序列是否已经出现在训练数据中

## Step 5. 用 ProteinMPNN 打分

运行：

```bash
bash ProteinMPNN/run_pplu_2G6X_chainB_scoring.sh
```

作用：

- 基于给定 PDB 结构，对生成的候选序列进行结构一致性打分

注意：

- 你需要自行准备对应的 PDB 文件
- 你需要自行下载 ProteinMPNN 的模型权重
- 由于版权或分发限制，这里不直接提供对应预训练权重

## Step 6. 用 Venus-MAXWELL 打分

运行：

```bash
bash Venus-MAXWELL/run_pplu_2G6X_chainB_scoring.sh
```

作用：

- 对候选序列做进一步模型打分

注意：

- 需要你自行从 Venus-MAXWELL 原论文或官方提供地址下载模型
- 这里同样不直接附带模型权重

## 最终产物

整个流程结束后，你会得到：

- 5-fold 亮度预测模型
- 一批 steer-PLM 生成的候选序列
- ProteinMPNN 打分结果
- Venus-MAXWELL 打分结果

你可以综合亮度预测分数、ProteinMPNN 分数和 Venus-MAXWELL 分数，筛选最终用于实验验证的序列。
