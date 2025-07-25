#include "kernels.h"

#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

#include <cmath>


// Deprecated: Multihead Attention will not use this softmax kernel.

void kernel(softmax_arg_t *arg) {
  auto *x = reinterpret_cast<float *>(arg->x_addr);
  auto size = arg->size;
  auto elements_per_thread = arg->elements_per_thread;

  int num_threads = vx_num_threads() * vx_num_cores() * vx_num_warps();
  auto *max_buffer = (float *)__local_mem(num_threads * sizeof(float));
  auto *global_reduce = reinterpret_cast<float *>(arg->global_reduce_addr);
  auto *global_core_reduce =
      reinterpret_cast<float *>(arg->global_core_reduce_addr);

  int tid = blockIdx.y;
  int step = num_threads;

  // Find max value (for numerical stability)
  float max_val = tid < size ? x[tid] : 0;
  for (int i = tid; i < size; i += step) {
    if (x[i] > max_val)
      max_val = x[i];
  }

  max_buffer[tid] = max_val;
  __syncthreads();

  float core_max_val = max_buffer[0];
  if (vx_thread_id() == 0 && vx_warp_id() == 0) {
    for (int i = 0; i < vx_num_warps() * vx_num_threads(); ++i) {
      int offset = vx_core_id() * vx_num_warps() * vx_num_threads();
      if (max_buffer[offset + i] > core_max_val) {
        core_max_val = max_buffer[offset + i];
      }
    }
    global_core_reduce[vx_core_id()] = core_max_val;
  }

  __syncthreads();
  if (tid == 0) {
    core_max_val = global_core_reduce[0];
    for (int i = 0; i < vx_num_cores(); ++i) {
      if (global_core_reduce[i] > core_max_val) {
        max_val = global_core_reduce[i];
      }
    }

    *global_reduce = max_val;
  }

  // global barrier
  vx_barrier(0x80000000, vx_num_cores());
  max_val = *global_reduce;

  auto *sum_buffer = (float *)__local_mem(num_threads * sizeof(float));

  // exp and sum
  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }

  sum_buffer[tid] = sum;
  __syncthreads();

  float core_sum = 0.0f;
  if (vx_thread_id() == 0 && vx_warp_id() == 0) {
    for (int i = 0; i < vx_num_warps() * vx_num_threads(); ++i) {
      int offset = vx_core_id() * vx_num_warps() * vx_num_threads();
      core_sum += sum_buffer[offset + i];
    }
    global_core_reduce[vx_core_id()] = core_sum;
  }

  __syncthreads();
  if (tid == 0) {
    sum = 0.0f;

    for (int i = 0; i < vx_num_cores(); ++i) {
      sum += global_core_reduce[i];
      *global_reduce = sum;
    }
  }

  // global barrier
  vx_barrier(0x80000000, vx_num_cores());
  sum = *global_reduce;

  // normalize
  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}

int main() {
  auto *arg = (softmax_arg_t *)csr_read(VX_CSR_MSCRATCH);
  int total_threads = vx_num_cores() * vx_num_warps() * vx_num_threads();
  uint32_t grid_dim[2] = {1, (uint32_t)total_threads};

  return vx_spawn_threads(2, grid_dim, nullptr, (vx_kernel_func_cb)kernel, arg);
}