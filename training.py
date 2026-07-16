"""
训练模块 - 训练循环、前向/反向传播、损失计算
"""
import numpy as np
from config import LEARNING_RATE, VOCAB_SIZE, DTYPE
from model import SimpleTransformer

def compute_loss(logits, targets):
    """
    计算交叉熵损失
    logits: (batch_size, seq_len, vocab_size)
    targets: (batch_size, seq_len)
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # 计算交叉熵损失
    # 对于每个位置，取目标词的概率
    target_probs = np.zeros((batch_size, seq_len))
    for i in range(batch_size):
        for j in range(seq_len):
            target_probs[i, j] = probs[i, j, targets[i, j]]

    # 避免log(0)
    target_probs = np.clip(target_probs, 1e-10, 1.0)

    # 计算负对数似然
    loss = -np.mean(np.log(target_probs))

    return loss, probs

def compute_gradients(logits, targets, model):
    """
    计算梯度（简化版）
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # 损失对logits的梯度
    grad_logits = probs.copy()

    # 对于目标位置，减去1
    for i in range(batch_size):
        for j in range(seq_len):
            grad_logits[i, j, targets[i, j]] -= 1

    # 归一化
    grad_logits = grad_logits / batch_size

    return grad_logits

def train_step(model, inputs, targets, learning_rate=LEARNING_RATE):
    """
    单个训练步骤
    """
    # 前向传播
    logits = model.forward(inputs)

    # 计算损失
    loss, probs = compute_loss(logits, targets)

    # 计算梯度
    grad_logits = compute_gradients(logits, targets, model)

    # 反向传播
    model.backward(grad_logits)

    # 更新参数
    model.update(learning_rate)

    # 收集统计信息
    stats = {
        'loss': loss,
        'logits_mean': np.mean(logits),
        'logits_std': np.std(logits),
        'probs_mean': np.mean(probs),
        'probs_std': np.std(probs),
        'grad_mean': np.mean(grad_logits),
        'grad_std': np.std(grad_logits),
        'grad_max': np.max(grad_logits),
        'grad_min': np.min(grad_logits)
    }

    return stats, model.cache

def train_epoch(model, data_gen, steps, step_callback=None):
    """
    训练一个epoch
    """
    epoch_losses = []
    epoch_stats = []

    for step in range(steps):
        # 获取批次数据
        inputs, targets = next(data_gen)

        # 训练步骤
        stats, cache = train_step(model, inputs, targets)

        epoch_losses.append(stats['loss'])
        epoch_stats.append(stats)

        # 打印进度
        if (step + 1) % 10 == 0:
            avg_loss = np.mean(epoch_losses[-10:])
            print(f"Step {step + 1}/{steps}: Loss = {avg_loss:.4f}, "
                  f"Logits Mean = {stats['logits_mean']:.4f}, "
                  f"Grad Mean = {stats['grad_mean']:.6f}")

        # 回调函数（用于可视化）
        if step_callback is not None and (step + 1) % 10 == 0:
            activations = [cache[f'layer_{i}_output'] for i in range(len(model.layers))]
            step_callback(step + 1, stats['loss'], activations)

    return epoch_losses, epoch_stats

def evaluate_model(model, inputs, targets):
    """
    评估模型
    """
    # 前向传播
    logits = model.forward(inputs)

    # 计算损失
    loss, probs = compute_loss(logits, targets)

    # 计算准确率
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == targets)

    stats = {
        'loss': loss,
        'accuracy': accuracy,
        'perplexity': np.exp(loss)
    }

    return stats

def print_training_stats(stats, step):
    """
    打印训练统计信息
    """
    print(f"\n{'='*60}")
    print(f"Training Step {step} Statistics")
    print(f"{'='*60}")
    print(f"Loss: {stats['loss']:.6f}")
    print(f"Logits - Mean: {stats['logits_mean']:.6f}, Std: {stats['logits_std']:.6f}")
    print(f"Probs - Mean: {stats['probs_mean']:.6f}, Std: {stats['probs_std']:.6f}")
    print(f"Gradients - Mean: {stats['grad_mean']:.6f}, Std: {stats['grad_std']:.6f}")
    print(f"Gradients - Max: {stats['grad_max']:.6f}, Min: {stats['grad_min']:.6f}")
    print(f"{'='*60}\n")

def check_gradient_health(grad_mean, grad_std, grad_max, grad_min):
    """
    检查梯度健康状况
    """
    warnings = []

    if abs(grad_mean) > 1.0:
        warnings.append(f"梯度均值过大: {grad_mean:.6f}")

    if grad_std > 10.0:
        warnings.append(f"梯度标准差过大: {grad_std:.6f}")

    if abs(grad_max) > 100.0:
        warnings.append(f"梯度最大值过大: {grad_max:.6f}")

    if abs(grad_min) > 100.0:
        warnings.append(f"梯度最小值过大: {grad_min:.6f}")

    if grad_std < 1e-6:
        warnings.append(f"梯度标准差过小，可能梯度消失: {grad_std:.6f}")

    return warnings

def train_model(model, data_gen, n_epochs, steps_per_epoch, viz_callback=None):
    """
    完整训练流程
    """
    all_losses = []
    all_stats = []

    print(f"开始训练: {n_epochs} epochs, {steps_per_epoch} steps per epoch")
    print(f"总参数量: {model.count_parameters():,}")
    print(f"数据类型: {model.dtype}")
    print(f"学习率: {LEARNING_RATE}")
    print("-" * 60)

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        print("-" * 60)

        epoch_losses, epoch_stats = train_epoch(
            model, data_gen, steps_per_epoch, viz_callback
        )

        all_losses.extend(epoch_losses)
        all_stats.extend(epoch_stats)

        # Epoch统计
        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch + 1} 完成. 平均 Loss: {avg_loss:.4f}")

        # 检查梯度健康状况
        last_stats = epoch_stats[-1]
        warnings = check_gradient_health(
            last_stats['grad_mean'],
            last_stats['grad_std'],
            last_stats['grad_max'],
            last_stats['grad_min']
        )

        if warnings:
            print("梯度健康警告:")
            for warning in warnings:
                print(f"  - {warning}")

    print("\n训练完成!")
    print(f"最终Loss: {all_losses[-1]:.4f}")
    print(f"平均Loss: {np.mean(all_losses):.4f}")

    return all_losses, all_stats
