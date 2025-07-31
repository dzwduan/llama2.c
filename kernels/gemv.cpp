#include "kernels.h"

#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

void kernel(gemv_arg_t *arg) {
  auto x = reinterpret_cast<float *>(arg->x_addr);
  auto w = reinterpret_cast<float *>(arg->w_addr);
  auto xout = reinterpret_cast<float *>(arg->xout_addr);
  auto size = arg->n;

  int row = blockIdx.x;

  float sum(0.0f);
  for (int e = 0; e < size; ++e)
    sum += w[row * size + e] * x[e];

  xout[row] = sum;
}

int main() {
  auto *arg = (gemv_arg_t *)csr_read(VX_CSR_MSCRATCH);
  uint32_t grid_dim[2] = {arg->d};

  return vx_spawn_threads(1, grid_dim, nullptr, (vx_kernel_func_cb)kernel, arg);
}