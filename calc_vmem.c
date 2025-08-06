#include <stdio.h>
#include <stdlib.h>

// 从模型代码中复制Config结构体定义
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

// 辅助函数，将bytes转换为MB
double bytes_to_mb(size_t bytes) {
    return (double)bytes / (1024.0 * 1024.0);
}

// 核心计算函数 (改进版)
void calculate_memory_requirements(Config* p) {
    printf("--- Calculating Memory Requirements for Config ---\n");
    printf("Dim: %d, Hidden Dim: %d, Layers: %d, Seq Len: %d\n", p->dim, p->hidden_dim, p->n_layers, p->seq_len);
    printf("Heads: %d, KV Heads: %d, Vocab Size: %d\n", p->n_heads, p->n_kv_heads, p->vocab_size);
    printf("--------------------------------------------------\n\n");

    // --- 1. 计算通用组件大小 (单位: bytes) ---

    // head_size 和 kv_dim 是理解结构的关键
    size_t head_size = p->dim / p->n_heads;
    size_t kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

    printf("Derived dimensions: head_size = %zu, kv_dim = %zu\n\n", head_size, kv_dim);

    // a. 模型权重大小
    size_t weights_size = 0;
    // 假设 wcls 不与 token_embedding_table 共享权重，以计算最大内存占用
    weights_size += (size_t)p->vocab_size * p->dim * sizeof(float); // token_embedding_table
    weights_size += (size_t)p->n_layers * p->dim * sizeof(float);   // rms_att_weight
    weights_size += (size_t)p->n_layers * p->dim * (p->n_heads * head_size) * sizeof(float); // wq
    weights_size += (size_t)p->n_layers * p->dim * kv_dim * sizeof(float); // wk
    weights_size += (size_t)p->n_layers * p->dim * kv_dim * sizeof(float); // wv
    weights_size += (size_t)p->n_layers * (p->n_heads * head_size) * p->dim * sizeof(float); // wo
    weights_size += (size_t)p->n_layers * p->dim * sizeof(float);   // rms_ffn_weight
    weights_size += (size_t)p->n_layers * p->dim * p->hidden_dim * sizeof(float); // w1
    weights_size += (size_t)p->n_layers * p->hidden_dim * p->dim * sizeof(float); // w2
    weights_size += (size_t)p->n_layers * p->dim * p->hidden_dim * sizeof(float); // w3
    weights_size += (size_t)p->dim * sizeof(float);                   // rms_final_weight
    weights_size += (size_t)p->vocab_size * p->dim * sizeof(float); // wcls

    // b. 运行时状态大小 (主要是KV Cache)
    size_t run_state_size = 0;
    run_state_size += (size_t)p->dim * sizeof(float); // x
    run_state_size += (size_t)p->dim * sizeof(float); // xb
    run_state_size += (size_t)p->dim * sizeof(float); // xb2
    run_state_size += (size_t)p->hidden_dim * sizeof(float); // hb
    run_state_size += (size_t)p->hidden_dim * sizeof(float); // hb2
    run_state_size += (size_t)p->dim * sizeof(float); // q
    run_state_size += (size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float); // key_cache
    run_state_size += (size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float); // value_cache
    run_state_size += (size_t)p->n_heads * p->seq_len * sizeof(float); // att
    run_state_size += (size_t)p->vocab_size * sizeof(float); // logits

    // c. CPU内存峰值估算
    size_t cpu_peak_memory = weights_size + run_state_size;
    
    printf("--- CPU Memory Estimation ---\n");
    printf("Model Weights Size: %.2f MB\n", bytes_to_mb(weights_size));
    printf("RunState Size (Activations + KV Cache): %.2f MB\n", bytes_to_mb(run_state_size));
    printf("Estimated Peak CPU RAM Usage: %.2f MB\n\n", bytes_to_mb(cpu_peak_memory));


    // --- 2. 计算GPGPU VRAM峰值 ---
    size_t v2_vram_peak = weights_size + run_state_size;
    printf("--- Version 2 (Pre-allocation) GPGPU VRAM Estimation ---\n");
    printf("Total VRAM usage (Weights + RunState are resident): %.2f MB\n\n", bytes_to_mb(v2_vram_peak));

    size_t v1_vram_peak = 0;
    size_t rmsnorm_temp = 3 * (size_t)p->dim * sizeof(float);
    if (rmsnorm_temp > v1_vram_peak) v1_vram_peak = rmsnorm_temp;
    
    size_t matmul_w1_temp = ((size_t)p->hidden_dim + p->dim + (size_t)p->dim * p->hidden_dim) * sizeof(float);
    if (matmul_w1_temp > v1_vram_peak) v1_vram_peak = matmul_w1_temp;
    
    // 检查 Classifier MatMul 是否更大
    size_t matmul_cls_temp = ((size_t)p->vocab_size + p->dim + (size_t)p->dim * p->vocab_size) * sizeof(float);
    if (matmul_cls_temp > v1_vram_peak) v1_vram_peak = matmul_cls_temp;

    printf("--- Version 1 (Just-in-Time) GPGPU VRAM Estimation ---\n");
    printf("Peak VRAM is determined by the largest single operation.\n");
    printf("Largest MatMul (Classifier) temp VRAM: %.2f MB\n", bytes_to_mb(matmul_cls_temp));
    printf("Estimated Peak VRAM Usage: %.2f MB\n\n", bytes_to_mb(v1_vram_peak));
}

int main() {
    Config model_config_from_image = {
        .dim = 288,
        .hidden_dim = 768,
        .n_layers = 6,
        .n_heads = 6,
        .n_kv_heads = 6,
        .vocab_size = 32000,
        .seq_len = 256
    };

    calculate_memory_requirements(&model_config_from_image);
    return 0;
}