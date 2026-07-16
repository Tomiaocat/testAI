"""
数据处理模块 - 从文件读取语料、分词、数据生成器
"""
import numpy as np
from collections import Counter
import re
from config import VOCAB_SIZE, SEQ_LENGTH, BATCH_SIZE, DTYPE, RANDOM_SEED, CORPUS_FILE

np.random.seed(RANDOM_SEED)

class SimpleTokenizer:
    """简单的分词器"""
    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_built = False

    def build_vocab(self, texts):
        """从文本构建词汇表"""
        # 统计词频
        word_freq = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_freq.update(words)

        # 选择最常见的词
        most_common = word_freq.most_common(self.vocab_size - 2)  # 保留PAD和UNK

        # 构建词汇表
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>'}

        for idx, (word, _) in enumerate(most_common, start=2):
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

        self.vocab_built = True
        print(f"词汇表构建完成，大小: {len(self.word_to_id)}")

    def _tokenize(self, text):
        """简单的分词函数"""
        # 转小写，按空格和标点分割
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words

    def encode(self, text):
        """将文本编码为ID序列"""
        if not self.vocab_built:
            raise ValueError("词汇表未构建，请先调用build_vocab()")

        words = self._tokenize(text)
        ids = [self.word_to_id.get(word, 1) for word in words]  # 1是UNK
        return ids

    def decode(self, ids):
        """将ID序列解码为文本"""
        if not self.vocab_built:
            raise ValueError("词汇表未构建，请先调用build_vocab()")

        words = [self.id_to_word.get(id, '<UNK>') for id in ids]
        return ' '.join(words)

def load_corpus(file_path):
    """从文件加载语料"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"从 {file_path} 加载了 {len(texts)} 行文本")
        return texts
    except FileNotFoundError:
        print(f"警告: 文件 {file_path} 不存在，使用示例文本")
        return [
            "hello world this is a test",
            "machine learning is fascinating",
            "neural networks learn patterns",
            "deep learning models are powerful",
            "artificial intelligence is evolving",
            "transformers changed natural language processing",
            "attention mechanisms are important",
            "gradient descent optimizes weights",
            "backpropagation computes gradients",
            "training data teaches the model"
        ]

def prepare_data(texts, tokenizer, seq_length=SEQ_LENGTH):
    """准备训练数据"""
    # 对所有文本进行编码
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text)
        all_ids.extend(ids)

    # 将长序列切分为固定长度的片段
    sequences = []
    for i in range(0, len(all_ids) - seq_length):
        sequences.append(all_ids[i:i+seq_length])

    print(f"生成了 {len(sequences)} 个训练序列")
    return np.array(sequences, dtype=np.int32)

def data_generator(sequences, batch_size=BATCH_SIZE, shuffle=True):
    """数据生成器 - 内存友好的批处理"""
    n_samples = len(sequences)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    while True:
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            # 输入是序列[:-1]，目标是序列[1:]
            batch_sequences = sequences[batch_indices]
            inputs = batch_sequences[:, :-1]
            targets = batch_sequences[:, 1:]

            yield inputs, targets

        if shuffle:
            np.random.shuffle(indices)

def create_sample_corpus(file_path="sample.txt"):
    """创建示例语料文件（如果不存在）"""
    sample_texts = [
        "hello world this is a test",
        "machine learning is fascinating",
        "neural networks learn patterns",
        "deep learning models are powerful",
        "artificial intelligence is evolving",
        "transformers changed natural language processing",
        "attention mechanisms are important",
        "gradient descent optimizes weights",
        "backpropagation computes gradients",
        "training data teaches the model",
        "the model learns from examples",
        "weights are updated during training",
        "loss function measures error",
        "optimization minimizes the loss",
        "forward propagation computes predictions",
        "backward propagation computes gradients",
        "learning rate controls step size",
        "batch size affects training speed",
        "epochs determine training duration",
        "regularization prevents overfitting"
    ]

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for text in sample_texts:
                f.write(text + '\n')
        print(f"示例语料文件已创建: {file_path}")
    except Exception as e:
        print(f"创建示例语料文件失败: {e}")
