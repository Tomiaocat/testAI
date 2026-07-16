"""
可视化模块 - 使用matplotlib绘制训练过程中的各种图表
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from config import SAVE_DIR, DTYPE

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_weight_distribution(weights, layer_name, save_path):
    """绘制权重分布直方图"""
    plt.figure(figsize=(10, 6))
    plt.hist(weights.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"{layer_name} Weight Distribution", fontsize=14)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_loss_curve(losses, save_path):
    """绘制训练loss曲线"""
    plt.figure(figsize=(12, 6))
    plt.plot(losses, linewidth=2, color='red')
    plt.title("Training Loss Curve", fontsize=14)
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_gradient_distribution(gradients, layer_name, save_path):
    """绘制梯度分布直方图"""
    plt.figure(figsize=(10, 6))
    plt.hist(gradients.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.title(f"{layer_name} Gradient Distribution", fontsize=14)
    plt.xlabel("Gradient Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_activation_heatmap(activations, layer_name, save_path):
    """绘制激活值热力图"""
    # 取第一个样本的激活值
    activation = activations[0]  # (seq_len, d_model)

    plt.figure(figsize=(12, 8))
    plt.imshow(activation.T, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Activation Value')
    plt.title(f"{layer_name} Activation Heatmap (First Sample)", fontsize=14)
    plt.xlabel("Sequence Position", fontsize=12)
    plt.ylabel("Hidden Dimension", fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_attention_weights(attention_weights, layer_name, head_idx=0, save_path=None):
    """绘制注意力权重热力图"""
    # attention_weights shape: (batch, n_heads, seq_len, seq_len)
    # 取第一个样本，指定头
    attn = attention_weights[0, head_idx]  # (seq_len, seq_len)

    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Attention Weight')
    plt.title(f"{layer_name} Attention Weights (Head {head_idx})", fontsize=14)
    plt.xlabel("Key Position", fontsize=12)
    plt.ylabel("Query Position", fontsize=12)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_parameter_evolution(param_history, layer_name, save_path):
    """绘制参数随训练的变化"""
    plt.figure(figsize=(12, 6))

    for i, (step, params) in enumerate(param_history):
        mean_val = np.mean(params)
        std_val = np.std(params)
        plt.scatter(step, mean_val, alpha=0.6, label=f'Step {step}' if i < 5 else None)

    plt.title(f"{layer_name} Parameter Evolution (Mean)", fontsize=14)
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Parameter Mean", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_summary(losses, param_stats, save_path):
    """绘制训练摘要图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Loss曲线
    axes[0, 0].plot(losses, linewidth=2, color='red')
    axes[0, 0].set_title("Training Loss", fontsize=12)
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # 参数均值变化
    if param_stats:
        steps = list(param_stats.keys())
        embedding_means = [param_stats[s]['embedding_mean'] for s in steps]
        axes[0, 1].plot(steps, embedding_means, marker='o', linewidth=2)
        axes[0, 1].set_title("Embedding Mean Evolution", fontsize=12)
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Mean Value")
        axes[0, 1].grid(True, alpha=0.3)

    # 参数标准差变化
    if param_stats:
        embedding_stds = [param_stats[s]['embedding_std'] for s in steps]
        axes[1, 0].plot(steps, embedding_stds, marker='s', linewidth=2, color='green')
        axes[1, 0].set_title("Embedding Std Evolution", fontsize=12)
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Std Value")
        axes[1, 0].grid(True, alpha=0.3)

    # 梯度统计（如果有）
    axes[1, 1].text(0.5, 0.5, 'Gradient Statistics\n(To be implemented)',
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].set_title("Gradient Statistics", fontsize=12)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_model_init(model, save_dir=SAVE_DIR):
    """可视化模型初始化状态"""
    print("正在可视化模型初始化状态...")

    # 嵌入层权重分布
    plot_weight_distribution(
        model.embedding.weights,
        "Embedding Layer",
        os.path.join(save_dir, "weight_init_embedding.png")
    )

    # 输出层权重分布
    plot_weight_distribution(
        model.output_weight,
        "Output Layer",
        os.path.join(save_dir, "weight_init_output.png")
    )

    # 各层的权重分布
    for i, layer in enumerate(model.layers):
        # 注意力权重
        plot_weight_distribution(
            layer.self_attn.W_q,
            f"Layer {i} - Attention W_q",
            os.path.join(save_dir, f"weight_init_layer{i}_W_q.png")
        )

        # FFN权重
        plot_weight_distribution(
            layer.ffn.W1,
            f"Layer {i} - FFN W1",
            os.path.join(save_dir, f"weight_init_layer{i}_FFN_W1.png")
        )

    print(f"初始化可视化完成，保存在 {save_dir}")

def visualize_training_step(model, step, loss, activations, save_dir=SAVE_DIR):
    """可视化单个训练步骤"""
    print(f"正在可视化训练步骤 {step}...")

    # 保存loss
    if not hasattr(visualize_training_step, 'losses'):
        visualize_training_step.losses = []
    visualize_training_step.losses.append(loss)

    # 绘制loss曲线
    plot_loss_curve(
        visualize_training_step.losses,
        os.path.join(save_dir, f"loss_curve_step{step}.png")
    )

    # 可视化激活值
    if activations is not None:
        for i, activation in enumerate(activations):
            plot_activation_heatmap(
                activation,
                f"Layer {i}",
                os.path.join(save_dir, f"activation_step{step}_layer{i}.png")
            )

    print(f"步骤 {step} 可视化完成")

def print_parameter_stats(stats):
    """打印参数统计信息"""
    print("\n" + "="*50)
    print("参数统计信息")
    print("="*50)

    for layer_name, layer_stats in stats.items():
        print(f"\n{layer_name}:")
        for param_name, value in layer_stats.items():
            if isinstance(value, dict):
                print(f"  {param_name}:")
                for k, v in value.items():
                    if isinstance(v, (int, tuple, np.ndarray)):
                        print(f"    {k}: {v}")
                    elif isinstance(v, (float, np.floating)):
                        print(f"    {k}: {v:.6f}")
                    else:
                        print(f"    {k}: {v}")
            elif isinstance(value, (int, tuple, np.ndarray)):
                print(f"  {param_name}: {value}")
            elif isinstance(value, (float, np.floating)):
                print(f"  {param_name}: {value:.6f}")
            else:
                print(f"  {param_name}: {value}")

    print("="*50 + "\n")
