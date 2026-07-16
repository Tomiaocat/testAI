# 在2GB内存下模拟大模型训练

这是一个教学项目，用于在有限内存（2GB）环境下模拟大模型的训练过程，帮助理解神经网络训练的基本原理。

通过这个项目，你可以直观地看到：
- 模型权重如何随机初始化
- 数据如何经过前向传播
- 梯度如何反向传播
- 训练过程中参数的变化

## 功能特点

- ✅ 随机初始化权重数组
- ✅ 从文件读取语料进行训练
- ✅ 简化版Transformer架构（包含自注意力机制）
- ✅ 前向传播和反向传播可视化
- ✅ 图形化显示训练过程（权重分布、loss曲线、激活值热力图）
- ✅ 内存优化设计（使用float32、分批处理）
- ✅ 详细的训练统计信息

## 项目结构

```
testAI/
├── config.py         # 配置参数（模型大小、训练参数等）
── data.py           # 数据处理（分词、数据生成器）
├── model.py          # 模型定义（简化Transformer）
├── training.py       # 训练循环、梯度计算
├── visualize.py      # 可视化工具（matplotlib图表）
├── main.py           # 主入口文件
├── test_simple.py    # 简化测试脚本
├── requirements.txt  # 依赖项
├── sample.txt        # 示例语料（自动生成）
└── visualizations/   # 可视化结果目录
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
python main.py
```

程序会自动：
1. 创建示例语料文件 `sample.txt`
2. 构建词汇表和分词器
3. 初始化简化Transformer模型
4. 可视化初始化权重分布
5. 开始训练并实时显示进度
6. 生成训练过程中的可视化图表

## 自定义语料

将你自己的文本数据保存为 `sample.txt`，每行一个句子或段落：

```
hello world this is a test
machine learning is fascinating
neural networks learn patterns
```

然后运行：
```bash
python main.py
```

## 调整配置

编辑 `config.py` 文件来调整模型和训练参数：

```python
# 模型参数
VOCAB_SIZE = 1000          # 词汇表大小
D_MODEL = 128              # 模型维度（词向量维度）
N_LAYERS = 2               # Transformer层数
D_FF = 512                 # 前馈网络维度
SEQ_LENGTH = 64            # 序列长度

# 训练参数
BATCH_SIZE = 32            # 批处理大小
LEARNING_RATE = 0.001      # 学习率
N_EPOCHS = 10              # 训练轮数
STEPS_PER_EPOCH = 100      # 每轮训练步数

# 内存优化
DTYPE = np.float32         # 数据类型（float16更省内存）
```

## 模型架构

```
输入 (batch, seq_len)
    ↓
嵌入层 Embedding
    ↓ 将词ID映射为128维向量
    ↓ 输出: (batch, seq_len, 128)
    ↓
Transformer层 × N_LAYERS
    ↓
    ├── 自注意力 Self-Attention
    │   Q = X·W_q  (查询)
    │   K = X·W_k  (键)
    │   V = X·W_v  (值)
    │   scores = Q·K^T / √d  (注意力分数)
    │   weights = softmax(scores)  (注意力权重)
    │   output = weights·V  (加权求和)
    │
    ── 前馈网络 FFN
        hidden = ReLU(X·W1 + b1)
        output = hidden·W2 + b2
    ↓
输出层 Linear
    ↓ 投影到词汇表维度
    ↓ 输出: (batch, seq_len, vocab_size)
```

**参数量：** ~412K 参数
**内存占用：** ~3.58 MB（远低于2GB限制）

## 输出说明

### 控制台输出

- **初始化阶段**：显示模型参数统计信息
- **训练过程**：实时显示loss、梯度统计
- **训练完成**：显示最终参数对比和内存使用总结

### 可视化文件

所有可视化结果保存在 `visualizations/` 目录：

| 文件名 | 说明 |
|--------|------|
| `weight_init_embedding.png` | 嵌入层权重分布 |
| `weight_init_output.png` | 输出层权重分布 |
| `weight_init_layer{0,1}_W_q.png` | 注意力权重分布 |
| `weight_init_layer{0,1}_FFN_W1.png` | 前馈网络权重分布 |
| `loss_curve_step*.png` | 训练loss曲线 |
| `activation_step*_layer*.png` | 各层激活值热力图 |
| `training_stats.txt` | 训练统计信息 |

## 内存优化策略

1. **数据类型优化**：使用 `float32` 而非 `float64`
2. **分批处理**：避免一次性加载全部数据
3. **生成器模式**：按需生成数据，减少内存占用
4. **简化架构**：使用小模型（~412K参数）

## 核心概念

### 前向传播

数据从输入流向输出的过程：
```
输入文本 → 分词 → 嵌入 → 注意力 → FFN → 输出概率 → 预测
```

### 反向传播

从输出误差倒推更新权重的过程：
```
预测错误 → 计算梯度 → 反向传播 → 更新权重
```

### 自注意力机制

让模型理解词与词之间的关系：
```
"我喜欢吃苹果"
  ↓
自注意力让模型知道：
- "吃" 应该关注 "苹果"（宾语）
- "我" 应该关注 "吃"（主语）
```

## 故障排除

### 内存不足

```python
# 减小模型规模
D_MODEL = 64       # 原来是128
N_LAYERS = 1       # 原来是2

# 或者使用更小的数据类型
DTYPE = np.float16
```

### 训练不稳定

```python
# 降低学习率
LEARNING_RATE = 0.0001

# 减小批处理大小
BATCH_SIZE = 16
```

## 教学目的

本项目主要用于：

- 理解Transformer的基本架构
- 观察权重初始化的影响
- 学习前向传播和反向传播
- 理解梯度计算和参数更新
- 掌握内存优化技巧

## 注意事项

- 这是一个简化的教学模型，不是生产级别的实现
- 梯度计算使用了简化版本（用于教学目的）
- 仅适用于CPU环境，GPU加速需要额外配置
- 建议在Python 3.7+环境下运行

## 进一步探索

1. **观察权重变化**：查看 `visualizations/` 中的图表
2. **修改配置**：编辑 `config.py` 调整参数
3. **使用自己的语料**：替换 `sample.txt`
4. **完善梯度计算**：尝试实现精确的梯度推导

## 许可证

MIT License
