#include "kernels.h"

#include <vx_intrinsics.h>
#include <vx_spawn.h>

#include <cmath>

void kernel(swiglu_arg_t *arg) {
  auto *hb = reinterpret_cast<float *>(arg->hb_addr);
  auto *hb2 = reinterpret_cast<float *>(arg->hb2_addr);
  auto hidden_dim = arg->hidden_dim;

  int tid = blockIdx.y;

  float val = hb[tid];
  // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
  val *= (1.0f / (1.0f + expf(-val)));
  // elementwise multiply with w3(x)
  val *= hb2[tid];
  hb[tid] = val;
}

int main() {
  auto *arg = (swiglu_arg_t *)csr_read(VX_CSR_MSCRATCH);
  uint32_t grid_dim[2] = {1, arg->hidden_dim};

  return vx_spawn_threads(2, grid_dim, nullptr, (vx_kernel_func_cb)kernel, arg);
}