#include "kernels.h"

#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

#include <math.h>

void kernel(rmsnorm_arg_t *arg) {
  int num_threads = vx_num_cores() * vx_num_warps() * vx_num_threads();
  auto *sum_buffer = (float *)__local_mem(num_threads * sizeof(float));

  auto *x = reinterpret_cast<float *>(arg->x_addr);
  volatile auto *o = reinterpret_cast<float *>(arg->o_addr);
  auto *w = reinterpret_cast<float *>(arg->w_addr);
  volatile auto *global_ss = reinterpret_cast<float *>(arg->global_ss_addr);
  volatile auto *global_sums = reinterpret_cast<float *>(arg->global_sums_addr);
  auto size = arg->size;
  auto elements_per_thread = arg->elements_per_thread;

  int thread_id = blockIdx.y;

  // Step 1: thread-local partial sum
  float ss = 0.0f;
  for (int i = 0; i < elements_per_thread; ++i) {
    int j = thread_id + i * num_threads;
    if (j < size) {
      ss += x[j] * x[j];
    }
  }

  sum_buffer[thread_id] = ss;
  __syncthreads();

  // Step 2
  float core_sum = 0.0f;
  if (vx_thread_id() == 0 && vx_warp_id() == 0) {
    for (int i = 0; i < vx_num_warps() * vx_num_threads(); ++i) {
      int offset = vx_core_id() * vx_num_warps() * vx_num_threads();
      core_sum += sum_buffer[offset + i];
    }
    global_sums[vx_core_id()] = core_sum;
  }
  __syncthreads();

  if (thread_id == 0) {
    ss = 0.0f;
    for (int i = 0; i < vx_num_cores(); ++i) {
      ss += global_sums[i];
    }
  }

  if (thread_id == 0) {
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    *global_ss = ss;
  }

  // global barrier
  vx_barrier(0x80000000, vx_num_cores());
  ss = *global_ss;
  __syncthreads();

  for (int i = 0; i < elements_per_thread; ++i) {
    int j = thread_id + i * num_threads;
    if (j < size) {
      o[j] = x[j] * ss * w[j];
    }
  }
}

int main() {
  auto *arg = (rmsnorm_arg_t *)csr_read(VX_CSR_MSCRATCH);
  int total_threads = vx_num_threads() * vx_num_warps() * vx_num_cores();
  uint32_t grid_dim[2] = {1, (uint32_t)total_threads};

  return vx_spawn_threads(2, grid_dim, nullptr, (vx_kernel_func_cb)kernel, arg);
}