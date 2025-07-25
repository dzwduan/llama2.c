#include "kernels.h"

#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

#include <cmath>

// Different from softmax.cpp
void softmax(float *x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;

  float *shared_buffer =
      (float *)__local_mem(sizeof(float) * vx_num_warps() * vx_num_threads());

  // find max value (for numerical stability)
  float max_val = tid < size ? x[tid] : 0;
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val)
      max_val = x[i];
  }

  shared_buffer[tid] = max_val;
  __syncthreads();

  if (tid == 0) {
    max_val = shared_buffer[0];
    for (int i = 0; i < vx_num_warps() * vx_num_threads(); i++) {
      if (shared_buffer[i] > max_val)
        max_val = shared_buffer[i];
    }

    shared_buffer[0] = max_val;
  }

  __syncthreads();
  max_val = shared_buffer[0];

  // exp and sum
  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }

  shared_buffer[tid] = sum;
  __syncthreads();

  if (tid == 0) {
    sum = 0.0f;
    for (int i = 0; i < vx_num_warps() * vx_num_threads(); i++) {
      sum += shared_buffer[i];
    }
    shared_buffer[0] = sum;
  }

  __syncthreads();
  sum = shared_buffer[0];

  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}

void kernel(attention_arg_t *arg) {
  auto *sxb = reinterpret_cast<float *>(arg->sxb_addr);
  auto *sq = reinterpret_cast<float *>(arg->sq_addr);
  auto *satt = reinterpret_cast<float *>(arg->satt_addr);
  auto *key_cache = reinterpret_cast<float *>(arg->key_cache_addr);
  auto *val_cache = reinterpret_cast<float *>(arg->value_cache_addr);

  int head_size = arg->head_size;
  int n_heads = arg->n_heads;
  int seq_len = arg->seq_len;
  int kv_dim = arg->kv_dim;
  int kv_mul = arg->kv_mul;
  int pos = arg->pos;
  int loff = arg->loff;

  int h = blockIdx.x;

  // get the query vector for this head
  float *q = sq + h * head_size;
  // attention scores for this head
  float *att = satt + h * seq_len;
  // iterate over all timesteps, including the current one
  for (int t = 0; t <= pos; t++) {
    // get the key vector for this head and at this timestep
    float *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
    // calculate the attention score as the dot product of q and k
    float score = 0.0f;
    for (int i = 0; i < head_size; i++) {
      score += q[i] * k[i];
    }
    score /= sqrtf(head_size);
    // save the score to the attention buffer
    att[t] = score;
    // vx_printf("att[%d][%d] = %f\n", h, t, score);
  }

  __syncthreads();

  softmax(att, pos + 1);

  // global barrier
  __syncthreads();

  // weighted sum of the values, store back into xb
  float *xb = sxb + h * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float val = 0.0f;
    for (int t = 0; t <= pos; t++) {
      // get the value vector for this head and at this timestep
      float *v = val_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
      // get the attention weight for this timestep
      float a = att[t];
      val += a * v[i];
    }
    xb[i] = val;
  }
}

int main() {
  auto *arg = (attention_arg_t *)csr_read(VX_CSR_MSCRATCH);
  uint32_t grid_dim[2] = {arg->n_heads, 1};

  uint32_t active_threads = vx_num_threads() * vx_num_warps();
  uint32_t block_dim[2] = {active_threads, 1};

  return vx_spawn_threads(2, grid_dim, block_dim, (vx_kernel_func_cb)kernel,
                          arg);
}