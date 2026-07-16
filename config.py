"""
配置参数 - 在2GB内存下模拟大模型训练
"""
import numpy as np

# 模型参数
VOCAB_SIZE = 1000          # 词汇表大小
D_MODEL = 128              # 模型维度
N_LAYERS = 2               # 层数
D_FF = 512                 # 前馈网络维度
SEQ_LENGTH = 64            # 序列长度

# 训练参数
BATCH_SIZE = 32            # 批处理大小
LEARNING_RATE = 0.001      # 学习率
N_EPOCHS = 2               # 训练轮数
STEPS_PER_EPOCH = 20       # 每轮训练步数

# 内存优化参数
DTYPE = np.float32         # 数据类型（float16更省内存但可能影响精度）
USE_GRADIENT_CHECKPOINTING = False  # 梯度检查点（进一步节省内存）

# 可视化参数
VIZ_INTERVAL = 5           # 可视化间隔步数
SAVE_DIR = "visualizations"  # 可视化结果保存目录

# 文件路径
CORPUS_FILE = "sample.txt"  # 语料文件路径

# 随机种子
RANDOM_SEED = 42
