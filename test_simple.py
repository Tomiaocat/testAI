"""
简化测试版本 - 快速验证功能
"""
import numpy as np
from config import *
from data import load_corpus, SimpleTokenizer, prepare_data, data_generator, create_sample_corpus
from model import SimpleTransformer
from visualize import visualize_model_init, print_parameter_stats

def main():
    print("="*60)
    print("简化测试版本 - 快速验证")
    print("="*60)

    # 1. 准备数据
    print("\n[1/4] 准备数据...")
    create_sample_corpus(CORPUS_FILE)
    texts = load_corpus(CORPUS_FILE)

    tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.build_vocab(texts)

    sequences = prepare_data(texts, tokenizer, SEQ_LENGTH)
    print(f"训练数据形状: {sequences.shape}")

    # 2. 初始化模型
    print("\n[2/4] 初始化模型...")
    model = SimpleTransformer(
        vocab_size=len(tokenizer.word_to_id),
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dtype=DTYPE
    )

    param_stats = model.get_parameter_stats()
    print_parameter_stats(param_stats)
    print(f"总参数量: {model.count_parameters():,}")

    # 3. 可视化初始化
    print("\n[3/4] 可视化初始化状态...")
    visualize_model_init(model, SAVE_DIR)

    # 4. 简单前向传播测试
    print("\n[4/4] 测试前向传播...")
    data_gen = data_generator(sequences, batch_size=2, shuffle=False)
    inputs, targets = next(data_gen)

    print(f"输入形状: {inputs.shape}")
    print(f"输入维度: {inputs.ndim}")
    print(f"目标形状: {targets.shape}")

    # 前向传播
    try:
        logits = model.forward(inputs)
        print(f"输出形状: {logits.shape}")
        print(f"输出统计: mean={np.mean(logits):.4f}, std={np.std(logits):.4f}")
    except Exception as e:
        print(f"前向传播错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 计算简单损失
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    batch_indices = np.arange(logits.shape[0])[:, None]
    seq_indices = np.arange(logits.shape[1])[None, :]
    target_probs = probs[batch_indices, seq_indices, targets]
    target_probs = np.clip(target_probs, 1e-10, 1.0)
    loss = -np.mean(np.log(target_probs))

    print(f"交叉熵损失: {loss:.4f}")

    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
    print(f"可视化文件保存在: {SAVE_DIR}")

if __name__ == "__main__":
    main()
