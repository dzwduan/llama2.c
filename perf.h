#ifndef PERF_H
#define PERF_H

#include <time.h>
#include <string.h> // For memset
#include <stdio.h>  // For snprintf

// Data structure to hold performance metrics for an operator
typedef struct {
    long long total_time_us;
    char dim_info[32]; // To store formatted string like "[1, 288] x [288, 288]"
    int call_count;
} PerfData;

// Operator types for profiling, in execution order
typedef enum {
    OP_RMSNORM_ATT,
    OP_MATMUL_Q,
    OP_MATMUL_K,
    OP_MATMUL_V,
    OP_ROPE,
    OP_ATTENTION,
    OP_MATMUL_WO,
    OP_ACCUM_ATT,
    OP_RMSNORM_FFN,
    OP_MATMUL_W1,
    OP_MATMUL_W3,
    OP_SWIGLU,
    OP_MATMUL_W2,
    OP_ACCUM_FFN,
    OP_RMSNORM_FINAL,
    OP_MATMUL_CLS,
    NUM_OPS
} OpType;

// Operator names for reporting, matching the enum order
static const char* op_names[NUM_OPS] = {
    "RMSNorm (Att)",
    "MatMul (Q)",
    "MatMul (K)",
    "MatMul (V)",
    "RoPE",
    "Multi-Head Attention",
    "MatMul (WO)",
    "Accum (Attention)",
    "RMSNorm (FFN)",
    "MatMul (W1)",
    "MatMul (W3)",
    "SwiGLU",
    "MatMul (W2)",
    "Accum (FFN)",
    "RMSNorm (Final)",
    "MatMul (Classifier)"
};

// --- Helper Functions for Performance Analysis ---

// High-precision timer (returns time in microseconds)
static inline long long time_in_us() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000000LL + time.tv_nsec / 1000;
}

// Initializes the performance data array for a new position
static inline void init_perf_data(PerfData* data, int num_ops) {
    for (int i = 0; i < num_ops; i++) {
        data[i].total_time_us = 0;
        data[i].dim_info[0] = '\0';
        data[i].call_count = 0;
    }
}

// Records a single performance measurement
static inline void record_perf(PerfData* data, OpType op, long long start_us, long long end_us, const char* dim_str) {
    data[op].total_time_us += (end_us - start_us);
    data[op].call_count++;
    // Only store dimension string on the first call
    if (dim_str && data[op].dim_info[0] == '\0') {
        snprintf(data[op].dim_info, sizeof(data[op].dim_info), "%s", dim_str);
    }
}

#endif // PERF_H