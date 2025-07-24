#include "kernels.h"

#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

void kernel(accum_arg_t *arg) {
  auto *a = reinterpret_cast<float *>(arg->a_addr);
  auto *b = reinterpret_cast<float *>(arg->b_addr);
  uint32_t size = arg->size;

  int tid = blockIdx.y;

  if (tid < size)
    a[tid] += b[tid];
}

int main() {
  auto *arg = (accum_arg_t *)csr_read(VX_CSR_MSCRATCH);
  uint32_t grid_dim[2] = {1, arg->size};

  return vx_spawn_threads(2, grid_dim, nullptr, (vx_kernel_func_cb)kernel, arg);
}