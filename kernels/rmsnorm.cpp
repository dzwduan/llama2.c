#include "kernels.h"

#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

#include <cmath>

void kernel(rmsnorm_arg_t *arg) {
  int num_threads = vx_num_warps() * vx_num_threads();
  auto *sum_buffer = (float *)__local_mem(num_threads * sizeof(float));

  auto *x = reinterpret_cast<float *>(arg->x_addr);
  auto *o = reinterpret_cast<float *>(arg->o_addr);
  auto *w = reinterpret_cast<float *>(arg->w_addr);
  auto size = arg->size;
  auto elements_per_thread = arg->elements_per_thread;

  int tid = threadIdx.x;

  // Step 1: thread-local partial sum
  float ss = 0.0f;
  for (int i = 0; i < elements_per_thread; ++i) {
    int j = tid + i * num_threads;
    if (j < size) {
      ss += x[j] * x[j];
    }
  }

  sum_buffer[tid] = ss;
  __syncthreads();

  if (tid == 0) {
    ss = 0.0f;
    for (int i = 0; i < num_threads; ++i) {
      ss += sum_buffer[i];
    }
  }

  if (tid == 0) {
    ss /= size;
    ss += 1e-5f; // epsilon
    ss = 1.0f / sqrtf(ss);

    sum_buffer[0] = ss;
  }

  __syncthreads();
  ss = sum_buffer[0];

  for (int i = 0; i < elements_per_thread; ++i) {
    int j = tid + i * num_threads;
    if (j < size) {
      o[j] = x[j] * ss * w[j];
    }
  }
}

int main() {
  auto *arg = (rmsnorm_arg_t *)csr_read(VX_CSR_MSCRATCH);
  uint32_t grid_dim[2] = {1, 1};

  int total_core_threads = vx_num_warps() * vx_num_threads();
  uint32_t block_dim[2] = {(uint32_t)total_core_threads, 1};

  return vx_spawn_threads(2, grid_dim, block_dim, (vx_kernel_func_cb)kernel,
                          arg);
}