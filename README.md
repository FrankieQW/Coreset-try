# 题目二：ALOHA Sim Transfer Cube 基准测试

本目录实现了题目二第一步 Baseline：在 ALOHA Sim Transfer Cube Human Demonstrations 数据集上随机抽取 10% 轨迹，使用冻结的 ResNet-18 离线提取图像特征，将 `[视觉特征 + 语言指令]` 回归到单臂 `7` 自由度动作，并报告动作预测 MSE。

## Conda 环境

建议使用 Python 3.10 或 3.11。下面以 CUDA 12.1 为例；如果没有 NVIDIA GPU，可以把 PyTorch 安装命令换成 CPU 版本。

```bash
conda create -n rzgc2 python=3.10 -y
conda activate rzgc2

pip install pandas pyarrow numpy tqdm pytest
conda install -c conda-forge av -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

如果本机没有 GPU：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 数据位置

代码默认读取当前目录下已经下载好的数据集：

```text
aloha_sim_transfer_cube_human/
```

当前数据集包含 `50` 条 episode、`20000` 帧，图像使用 `observation.images.top` 视角。数据的 `action` 是双臂 `14` 维，本 baseline 默认取右臂后 `7` 维；如需左臂可使用 `--arm left`。

## 运行基准测试

```bash
python run_baseline.py
```

常用参数：

```bash
python run_baseline.py \
  --dataset-dir aloha_sim_transfer_cube_human \
  --output-dir outputs/baseline_random10_resnet18 \
  --arm right \
  --sample-fraction 0.10 \
  --epochs 80 \
  --batch-size 64 \
  --instruction "Transfer the cube to the target position."
```

运行后会输出每轮训练的 `train_mse` 和 `val_mse`，最终结果保存在：

```text
outputs/baseline_random10_resnet18/
  features_vision.npy
  targets_action.npy
  sampled_rows.csv
  mlp_action_regressor.pt
  baseline_results.json
```

报告中建议引用 `baseline_results.json` 里的 `metrics.best_val_mse` 或 `metrics.final_val_mse` 作为随机 10% baseline 的 MSE。

## 运行脑启发核心集验证

第二步使用独立脚本 `run_coreset.py`，不会改动或调用 baseline 代码。该脚本实现 `PD-Coreset`：根据动作变化、机械臂状态变化、视觉预测误差、全局特征稀缺性和时间抑制，从全量数据中筛选 10% 核心帧，再用结构相同的 MLP 重新训练并报告 MSE。

```bash
python run_coreset.py \
  --dataset-dir aloha_sim_transfer_cube_human \
  --output-dir outputs/coreset_pd10_resnet18 \
  --arm right \
  --coreset-fraction 0.10 \
  --epochs 80 \
  --batch-size 64 \
  --instruction "Transfer the cube to the target position."
```

如果某个输出目录里已经有全量 `features_vision.npy`，也可以用 `--baseline-output-dir` 复用特征，避免重复提取。注意第一步随机 baseline 默认只保存 10% 样本特征，不能作为全量特征复用。

核心集价值分数为：

```text
score = 0.35 * action_delta
      + 0.25 * state_delta
      + 0.25 * vision_delta
      + 0.15 * feature_rarity
```

其中 `action_delta`、`state_delta` 和 `vision_delta` 对应预测编码中的变化/预测误差，`feature_rarity` 用于降低分布冗余，`--temporal-window` 用于抑制相邻帧连续入选。

运行后结果保存在：

```text
outputs/coreset_pd10_resnet18/
  features_vision.npy
  targets_action.npy
  full_rows.csv
  coreset_indices.csv
  mlp_action_regressor.pt
  coreset_results.json
```

报告中建议引用 `coreset_results.json` 里的 `metrics.best_val_mse` 或 `metrics.final_val_mse`，并与随机 baseline 的 `baseline_results.json` 对比。

## 方法说明

1. 按 episode 随机抽取 10% 轨迹。对当前 50 条 episode，即抽取 5 条。
2. 使用冻结权重的 ImageNet 预训练 ResNet-18，去掉分类头后得到每帧 512 维视觉特征。
3. 语言指令使用轻量哈希 bag-of-words 编码为 128 维向量。由于本数据集只有一个任务指令，该编码在所有样本上保持一致，用于满足 `[视觉特征 + 语言指令]` 输入形式。
4. 将 `[512 维视觉特征 + 128 维语言特征]` 输入两层隐藏层 MLP，预测单臂 7 维动作。
5. 在抽中的 10% episode 内按 episode 做 80/20 训练/验证划分，并报告验证集 MSE。

## 快速测试

安装 `pytest` 后可运行：

```bash
python -m pytest tests/test_baseline_utils.py tests/test_coreset_utils.py
```

该测试只覆盖抽样、动作维度选择和语言特征构造，不需要安装 PyTorch。
