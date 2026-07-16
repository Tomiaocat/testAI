"""
主入口文件 - 在2GB内存下模拟大模型训练
"""
import numpy as np
import os
import sys
from config import *
from data import load_corpus, SimpleTokenizer, prepare_data, data_generator, create_sample_corpus
from model import SimpleTransformer
from training import train_model, print_training_stats, check_gradient_health
from visualize import visualize_model_init, visualize_training_step, print_parameter_stats

def main():
    print("="*60)
    print("在2GB内存下模拟大模型训练")
    print("="*60)
    print(f"配置信息:")
    print(f"  词汇表大小: {VOCAB_SIZE}")
    print(f"  模型维度: {D_MODEL}")
    print(f"  层数: {N_LAYERS}")
    print(f"  批处理大小: {BATCH_SIZE}")
    print(f"  序列长度: {SEQ_LENGTH}")
    print(f"  数据类型: {DTYPE}")
    print(f"  学习率: {LEARNING_RATE}")
    print("="*60)

    # 1. 准备数据
    print("\n[1/6] 准备数据...")
    create_sample_corpus(CORPUS_FILE)  # 创建示例语料（如果不存在）
    texts = load_corpus(CORPUS_FILE)

    # 构建分词器
    tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.build_vocab(texts)

    # 准备训练数据
    sequences = prepare_data(texts, tokenizer, SEQ_LENGTH)
    print(f"训练数据形状: {sequences.shape}")

    # 创建数据生成器
    data_gen = data_generator(sequences, BATCH_SIZE, shuffle=True)

    # 2. 初始化模型
    print("\n[2/6] 初始化模型...")
    model = SimpleTransformer(
        vocab_size=len(tokenizer.word_to_id),
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dtype=DTYPE
    )

    # 打印参数统计
    param_stats = model.get_parameter_stats()
    print_parameter_stats(param_stats)

    print(f"总参数量: {model.count_parameters():,}")

    # 估算内存使用
    param_memory = model.count_parameters() * np.dtype(DTYPE).itemsize / (1024**2)
    print(f"参数内存占用: {param_memory:.2f} MB")

    # 3. 可视化初始化状态
    print("\n[3/6] 可视化模型初始化状态...")
    visualize_model_init(model, SAVE_DIR)

    # 4. 定义可视化回调
    def viz_callback(step, loss, activations):
        if step % VIZ_INTERVAL == 0:
            visualize_training_step(model, step, loss, activations, SAVE_DIR)

            # 打印当前参数统计
            current_stats = model.get_parameter_stats()
            print(f"\n步骤 {step} 参数变化:")
            print(f"  Embedding Mean: {current_stats['embedding']['mean']:.6f}")
            print(f"  Embedding Std: {current_stats['embedding']['std']:.6f}")

    # 5. 训练模型
    print("\n[4/6] 开始训练...")
    print(f"训练配置: {N_EPOCHS} epochs, {STEPS_PER_EPOCH} steps per epoch")
    print(f"总训练步数: {N_EPOCHS * STEPS_PER_EPOCH}")

    try:
        all_losses, all_stats = train_model(
            model,
            data_gen,
            n_epochs=N_EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            viz_callback=viz_callback
        )
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        return
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. 训练后分析
    print("\n[5/6] 训练后分析...")

    # 最终参数统计
    final_stats = model.get_parameter_stats()
    print_parameter_stats(final_stats)

    # 比较初始化和训练后的参数
    print("\n参数变化对比:")
    print(f"Embedding Mean: {param_stats['embedding']['mean']:.6f} -> {final_stats['embedding']['mean']:.6f}")
    print(f"Embedding Std: {param_stats['embedding']['std']:.6f} -> {final_stats['embedding']['std']:.6f}")

    # 7. 生成最终可视化
    print("\n[6/6] 生成最终可视化...")
    visualize_model_init(model, SAVE_DIR)

    # 保存训练统计
    stats_file = os.path.join(SAVE_DIR, "training_stats.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("训练统计信息\n")
        f.write("="*60 + "\n")
        f.write(f"总训练步数: {len(all_losses)}\n")
        f.write(f"最终Loss: {all_losses[-1]:.6f}\n")
        f.write(f"平均Loss: {np.mean(all_losses):.6f}\n")
        f.write(f"最小Loss: {np.min(all_losses):.6f}\n")
        f.write(f"最大Loss: {np.max(all_losses):.6f}\n")
        f.write("\n配置参数:\n")
        f.write(f"词汇表大小: {VOCAB_SIZE}\n")
        f.write(f"模型维度: {D_MODEL}\n")
        f.write(f"前馈网络维度: {D_FF}\n")
        f.write(f"层数: {N_LAYERS}\n")
        f.write(f"批处理大小: {BATCH_SIZE}\n")
        f.write(f"学习率: {LEARNING_RATE}\n")
        f.write(f"数据类型: {DTYPE}\n")

    print(f"\n训练统计已保存到: {stats_file}")
    print(f"所有可视化结果保存在: {SAVE_DIR}")

    # 内存使用总结
    print("\n内存使用总结:")
    print(f"  参数内存: {param_memory:.2f} MB")
    print(f"  批处理数据内存: {BATCH_SIZE * SEQ_LENGTH * np.dtype(np.int32).itemsize / (1024**2):.2f} MB")
    print(f"  激活值内存 (估算): {N_LAYERS * BATCH_SIZE * SEQ_LENGTH * D_MODEL * np.dtype(DTYPE).itemsize / (1024**2):.2f} MB")
    print(f"  总内存占用 (估算): {param_memory + BATCH_SIZE * SEQ_LENGTH * np.dtype(np.int32).itemsize / (1024**2) + N_LAYERS * BATCH_SIZE * SEQ_LENGTH * D_MODEL * np.dtype(DTYPE).itemsize / (1024**2):.2f} MB")

    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)

if __name__ == "__main__":
    main()
