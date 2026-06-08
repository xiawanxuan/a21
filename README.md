# 基于Transformer的天文光谱红移自动测量与分类系统

## 项目简介

本项目基于Python和PyTorch构建，使用Transformer架构实现天文光谱的红移自动测量与天体分类系统。系统支持光谱数据预处理、Transformer特征提取、红移值回归预测、天体光谱分类、模型训练与评估、推理结果可视化等功能。

## 目录结构

```
.
├── config.py                  # 全局配置文件
├── main.py                    # 主入口文件
├── requirements.txt           # 依赖包列表
├── src/                       # 源码目录
│   ├── __init__.py
│   ├── data_processing.py     # 数据预处理模块
│   ├── models.py              # 模型构建模块
│   ├── train_eval.py          # 训练评估模块
│   └── inference_visualization.py  # 推理可视化模块
├── scripts/                   # 脚本目录
│   └── generate_sample_data.py    # 示例数据生成脚本
├── data/                      # 数据目录
├── models/                    # 模型保存目录
├── results/                   # 训练结果目录
└── outputs/                   # 输出结果目录
```

## 模块说明

### 1. 数据预处理模块 (data_processing.py)
- 光谱数据加载与保存
- Z-score标准化与Min-Max归一化
- 数据增强（噪声添加、光谱位移、强度缩放）
- 数据集类与数据加载器构建

### 2. 模型构建模块 (models.py)
- 位置编码 (Positional Encoding)
- 光谱嵌入层 (Spectrum Embedding)
- Transformer编码器
- 红移回归预测头
- 天体分类预测头
- 完整的SpectrumTransformer模型

### 3. 训练评估模块 (train_eval.py)
- 组合损失函数（MSE回归损失 + 交叉熵分类损失）
- 训练循环与验证循环
- 评估指标：MAE、RMSE、σ_NI、异常率、准确率
- 学习率调度（Warmup + Cosine Annealing）
- 模型检查点保存与加载

### 4. 推理可视化模块 (inference_visualization.py)
- 单条光谱推理
- 批量光谱推理
- 光谱图绘制
- 红移预测对比图
- 残差分析图
- 混淆矩阵图
- 训练曲线图

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成示例数据

```bash
python main.py generate --num_samples 5000
```

### 3. 训练模型

```bash
python main.py train --epochs 50 --batch_size 32 --lr 1e-4
```

### 4. 评估模型

```bash
python main.py test --checkpoint ./models/best_model.pth
```

### 5. 批量推理

```bash
python main.py predict --checkpoint ./models/best_model.pth --input ./data/spectrum_data.npz --visualize
```

### 6. 生成可视化结果

```bash
python main.py visualize --checkpoint ./models/best_model.pth
```

## 自定义数据集训练

### 数据格式

数据集需保存为 `.npz` 格式，包含以下三个数组：
- `spectra`: 光谱数据，形状为 (N, spectrum_length)
- `redshifts`: 红移值，形状为 (N,)
- `labels`: 分类标签，形状为 (N,)，取值范围 0 到 num_classes-1

### 使用自定义数据训练

```bash
python main.py train --data_path /path/to/your/data.npz --epochs 50
```

## 配置说明

所有配置参数在 `config.py` 文件中，主要参数包括：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| spectrum_length | 1024 | 光谱长度 |
| num_classes | 5 | 天体类别数 |
| d_model | 128 | Transformer隐藏层维度 |
| nhead | 4 | 注意力头数 |
| num_encoder_layers | 4 | Transformer编码器层数 |
| dim_feedforward | 256 | 前馈网络维度 |
| batch_size | 32 | 批次大小 |
| num_epochs | 50 | 训练轮数 |
| learning_rate | 1e-4 | 学习率 |
| weight_decay | 1e-5 | 权重衰减 |

## 评估指标

系统使用以下指标评估模型性能：

**红移回归指标：**
- MAE (平均绝对误差)
- RMSE (均方根误差)
- σ_NI (归一化区间标准差，使用 |Δz|/(1+z) 的68.27百分位)
- 异常率 (|Δz|/(1+z) > 0.15 的比例)

**分类指标：**
- 准确率 (Accuracy)
- 混淆矩阵 (Confusion Matrix)
- 分类报告 (Precision, Recall, F1-score)

## 引用

如使用本项目，请参考：
- Transformer架构: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 光谱红移测量相关文献

## 许可证

MIT License
