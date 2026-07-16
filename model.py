"""
模型定义 - 简化版Transformer
包含：嵌入层、多头注意力、前馈网络、层归一化
"""
import numpy as np
from config import D_MODEL, N_LAYERS, D_FF, VOCAB_SIZE, DTYPE, RANDOM_SEED

np.random.seed(RANDOM_SEED)

class LayerNorm:
    """层归一化"""
    def __init__(self, d_model, dtype=DTYPE):
        self.d_model = d_model
        self.dtype = dtype
        self.gamma = np.ones(d_model, dtype=dtype)
        self.beta = np.zeros(d_model, dtype=dtype)
        self.cache = {}

    def forward(self, x, training=True):
        """前向传播"""
        self.cache['x'] = x

        # 计算均值和方差
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        # 归一化
        x_norm = (x - mean) / np.sqrt(var + 1e-6)

        # 缩放和平移
        output = self.gamma * x_norm + self.beta

        return output

    def backward(self, grad_output):
        """反向传播（简化版）"""
        x = self.cache['x']

        # 这里简化处理，实际需要完整的梯度计算
        # 为了教学目的，我们使用近似梯度
        grad_input = grad_output * self.gamma[np.newaxis, np.newaxis, :]

        # 更新参数的梯度
        self.grad_gamma = np.sum(grad_output, axis=(0, 1))
        self.grad_beta = np.sum(grad_output, axis=(0, 1))

        return grad_input

    def update(self, learning_rate):
        """更新参数"""
        self.gamma -= learning_rate * self.grad_gamma
        self.beta -= learning_rate * self.grad_beta

class Embedding:
    """嵌入层"""
    def __init__(self, vocab_size, d_model, dtype=DTYPE):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dtype = dtype

        # 随机初始化权重
        scale = np.sqrt(2.0 / d_model)
        self.weights = np.random.randn(vocab_size, d_model).astype(dtype) * scale

        self.cache = {}

    def forward(self, x):
        """前向传播"""
        self.cache['x'] = x

        # 手动构建embedding以确保正确的形状
        batch_size, seq_len = x.shape
        d_model = self.weights.shape[1]
        embedded = np.zeros((batch_size, seq_len, d_model), dtype=self.dtype)

        for i in range(batch_size):
            for j in range(seq_len):
                embedded[i, j] = self.weights[x[i, j]]

        return embedded

    def backward(self, grad_output):
        """反向传播"""
        x = self.cache['x']

        # 计算权重梯度
        grad_weights = np.zeros_like(self.weights)
        batch_size, seq_len = x.shape

        for i in range(batch_size):
            for j in range(seq_len):
                grad_weights[x[i, j]] += grad_output[i, j]

        self.grad_weights = grad_weights

        # 嵌入层的输入梯度通常不需要（输入是离散的）
        return None

    def update(self, learning_rate):
        """更新权重"""
        self.weights -= learning_rate * self.grad_weights

class SimpleAttention:
    """简化的自注意力机制"""
    def __init__(self, d_model, dtype=DTYPE):
        self.d_model = d_model
        self.dtype = dtype

        # 初始化Q、K、V的权重
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model).astype(dtype) * scale
        self.W_k = np.random.randn(d_model, d_model).astype(dtype) * scale
        self.W_v = np.random.randn(d_model, d_model).astype(dtype) * scale
        self.W_o = np.random.randn(d_model, d_model).astype(dtype) * scale

        self.cache = {}

    def forward(self, x):
        """前向传播"""
        self.cache['x'] = x

        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got {x.ndim}D with shape {x.shape}")

        batch_size, seq_len, d_model = x.shape

        # 计算Q、K、V
        Q = np.dot(x, self.W_q)  # (batch, seq_len, d_model)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # 计算注意力分数 - 使用einsum确保正确的矩阵乘法
        scores = np.einsum('bqd,bkd->bqk', Q, K) / np.sqrt(d_model)  # (batch, seq_len, seq_len)

        # Softmax
        attention_weights = self._softmax(scores)  # (batch, seq_len, seq_len)

        # 应用注意力权重 - 使用einsum确保正确的矩阵乘法
        output = np.einsum('bqk,bkd->bqd', attention_weights, V)  # (batch, seq_len, d_model)

        # 最终线性变换
        output = np.dot(output, self.W_o)  # (batch, seq_len, d_model)

        # 缓存用于反向传播
        self.cache.update({
            'Q': Q, 'K': K, 'V': V,
            'attention_weights': attention_weights,
            'output': output
        })

        return output

    def _softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def backward(self, grad_output):
        """反向传播（简化版）"""
        x = self.cache['x']

        # 简化的梯度计算
        grad_input = np.dot(grad_output, self.W_o.T)

        # 计算权重梯度（简化）- 使用正确的einsum
        batch_size, seq_len, d_model = x.shape
        self.grad_W_q = np.zeros_like(self.W_q)
        self.grad_W_k = np.zeros_like(self.W_k)
        self.grad_W_v = np.zeros_like(self.W_v)
        self.grad_W_o = np.zeros_like(self.W_o)

        for i in range(batch_size):
            for j in range(seq_len):
                self.grad_W_q += np.outer(x[i, j], grad_input[i, j])
                self.grad_W_k += np.outer(x[i, j], grad_input[i, j])
                self.grad_W_v += np.outer(x[i, j], grad_input[i, j])
                self.grad_W_o += np.outer(self.cache['output'][i, j], grad_output[i, j])

        # 平均
        self.grad_W_q /= (batch_size * seq_len)
        self.grad_W_k /= (batch_size * seq_len)
        self.grad_W_v /= (batch_size * seq_len)
        self.grad_W_o /= (batch_size * seq_len)

        return grad_input

    def update(self, learning_rate):
        """更新权重"""
        self.W_q -= learning_rate * self.grad_W_q
        self.W_k -= learning_rate * self.grad_W_k
        self.W_v -= learning_rate * self.grad_W_v
        self.W_o -= learning_rate * self.grad_W_o

class FeedForward:
    """前馈网络"""
    def __init__(self, d_model, d_ff, dtype=DTYPE):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dtype = dtype

        # 初始化权重
        scale = np.sqrt(2.0 / d_model)
        self.W1 = np.random.randn(d_model, d_ff).astype(dtype) * scale
        self.b1 = np.zeros(d_ff, dtype=dtype)
        self.W2 = np.random.randn(d_ff, d_model).astype(dtype) * scale
        self.b2 = np.zeros(d_model, dtype=dtype)

        self.cache = {}

    def forward(self, x):
        """前向传播"""
        self.cache['x'] = x

        # 确保输入是3D的
        if x.ndim != 3:
            raise ValueError(f"FFN expected 3D input, got {x.ndim}D with shape {x.shape}")

        # 第一层 + ReLU激活
        hidden = np.dot(x, self.W1) + self.b1  # b1会自动广播
        hidden = np.maximum(0, hidden)  # ReLU

        # 第二层
        output = np.dot(hidden, self.W2) + self.b2  # b2会自动广播

        self.cache['hidden'] = hidden
        return output

    def backward(self, grad_output):
        """反向传播"""
        x = self.cache['x']
        hidden = self.cache['hidden']

        batch_size, seq_len, d_model = x.shape
        d_ff = self.W1.shape[1]

        # 第二层梯度
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b2 = np.zeros_like(self.b2)

        for i in range(batch_size):
            for j in range(seq_len):
                self.grad_W2 += np.outer(hidden[i, j], grad_output[i, j])
                self.grad_b2 += grad_output[i, j]

        self.grad_W2 /= (batch_size * seq_len)
        self.grad_b2 /= (batch_size * seq_len)

        grad_hidden = np.dot(grad_output, self.W2.T)

        # ReLU梯度
        grad_hidden[hidden <= 0] = 0

        # 第一层梯度
        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_b1 = np.zeros_like(self.b1)

        for i in range(batch_size):
            for j in range(seq_len):
                self.grad_W1 += np.outer(x[i, j], grad_hidden[i, j])
                self.grad_b1 += grad_hidden[i, j]

        self.grad_W1 /= (batch_size * seq_len)
        self.grad_b1 /= (batch_size * seq_len)

        grad_input = np.dot(grad_hidden, self.W1.T)

        return grad_input

    def update(self, learning_rate):
        """更新权重"""
        self.W1 -= learning_rate * self.grad_W1
        self.b1 -= learning_rate * self.grad_b1
        self.W2 -= learning_rate * self.grad_W2
        self.b2 -= learning_rate * self.grad_b2

class TransformerLayer:
    """Transformer层"""
    def __init__(self, d_model, d_ff, dtype=DTYPE):
        self.self_attn = SimpleAttention(d_model, dtype)
        self.ffn = FeedForward(d_model, d_ff, dtype)

    def forward(self, x):
        """前向传播"""
        # 自注意力
        attn_output = self.self_attn.forward(x)

        # 前馈网络
        ffn_output = self.ffn.forward(attn_output)

        return ffn_output

    def backward(self, grad_output):
        """反向传播"""
        # 反向传播顺序与前向相反
        grad_x = self.ffn.backward(grad_output)
        grad_x = self.self_attn.backward(grad_x)

        return grad_x

    def update(self, learning_rate):
        """更新所有子层的参数"""
        self.self_attn.update(learning_rate)
        self.ffn.update(learning_rate)

class SimpleTransformer:
    """简化版Transformer模型"""
    def __init__(self, vocab_size, d_model=D_MODEL, n_layers=N_LAYERS, d_ff=D_FF, dtype=DTYPE):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dtype = dtype

        # 初始化各层
        self.embedding = Embedding(vocab_size, d_model, dtype)
        self.layers = [TransformerLayer(d_model, d_ff, dtype) for _ in range(n_layers)]

        # 输出层
        scale = np.sqrt(2.0 / d_model)
        self.output_weight = np.random.randn(d_model, vocab_size).astype(dtype) * scale

        self.cache = {}

    def forward(self, x):
        """前向传播"""
        self.cache['input'] = x

        # 嵌入
        x = self.embedding.forward(x)

        # 通过Transformer层
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            self.cache[f'layer_{i}_output'] = x

        # 输出投影
        logits = np.dot(x, self.output_weight)

        self.cache['logits'] = logits
        return logits

    def backward(self, grad_logits):
        """反向传播"""
        # 输出层梯度
        x = self.cache[f'layer_{self.n_layers-1}_output']
        batch_size, seq_len, d_model = x.shape
        vocab_size = self.output_weight.shape[1]

        # 使用循环计算梯度
        self.grad_output_weight = np.zeros_like(self.output_weight)
        for i in range(batch_size):
            for j in range(seq_len):
                self.grad_output_weight += np.outer(x[i, j], grad_logits[i, j])

        self.grad_output_weight /= (batch_size * seq_len)

        grad_x = np.dot(grad_logits, self.output_weight.T)

        # 通过Transformer层反向传播
        for i in reversed(range(self.n_layers)):
            grad_x = self.layers[i].backward(grad_x)

        # 嵌入层反向传播
        self.embedding.backward(grad_x)

        return grad_x

    def update(self, learning_rate):
        """更新所有参数"""
        for layer in self.layers:
            layer.update(learning_rate)

        self.embedding.update(learning_rate)

        # 更新输出权重
        self.output_weight -= learning_rate * self.grad_output_weight

    def get_parameter_stats(self):
        """获取参数统计信息"""
        stats = {
            'embedding': {
                'mean': np.mean(self.embedding.weights),
                'std': np.std(self.embedding.weights),
                'shape': self.embedding.weights.shape
            },
            'output_weight': {
                'mean': np.mean(self.output_weight),
                'std': np.std(self.output_weight),
                'shape': self.output_weight.shape
            }
        }

        for i, layer in enumerate(self.layers):
            stats[f'layer_{i}_attention'] = {
                'W_q_mean': np.mean(layer.self_attn.W_q),
                'W_q_std': np.std(layer.self_attn.W_q),
                'W_k_mean': np.mean(layer.self_attn.W_k),
                'W_k_std': np.std(layer.self_attn.W_k),
            }
            stats[f'layer_{i}_ffn'] = {
                'W1_mean': np.mean(layer.ffn.W1),
                'W1_std': np.std(layer.ffn.W1),
                'W2_mean': np.mean(layer.ffn.W2),
                'W2_std': np.std(layer.ffn.W2),
            }

        return stats

    def count_parameters(self):
        """计算参数总数"""
        total = 0
        total += self.embedding.weights.size
        total += self.output_weight.size

        for layer in self.layers:
            total += layer.self_attn.W_q.size
            total += layer.self_attn.W_k.size
            total += layer.self_attn.W_v.size
            total += layer.self_attn.W_o.size
            total += layer.ffn.W1.size
            total += layer.ffn.b1.size
            total += layer.ffn.W2.size
            total += layer.ffn.b2.size

        return total
