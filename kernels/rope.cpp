#include "kernels.h"

#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

#include <cmath>

void kernel(rope_arg_t *arg) {
  auto dim = arg->dim;
  auto head_size = arg->head_size;
  auto kv_dim = arg->kv_dim;
  auto pos = arg->pos;
  auto *q = reinterpret_cast<float *>(arg->q_addr);
  auto *k = reinterpret_cast<float *>(arg->k_addr);

  int i = blockIdx.y * 2;
  int head_dim = i % head_size;
  float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
  float val = pos * freq;
  float fcr = cosf(val);
  float fci = sinf(val);
  int rotn = i < kv_dim ? 2 : 1;

  for (int v = 0; v < rotn; ++v) {
    float *vec = v == 0 ? q : k; // the vector to rotate (query or key)
    float v0 = vec[i];
    float v1 = vec[i + 1];
    vec[i] = v0 * fcr - v1 * fci;
    vec[i + 1] = v0 * fci + v1 * fcr;
  }
}

int main() {
  auto *arg = (rope_arg_t *)csr_read(VX_CSR_MSCRATCH);
  uint32_t grid_dim[2] = {1, arg->dim / 2};

  return vx_spawn_threads(2, grid_dim, nullptr, (vx_kernel_func_cb)kernel, arg);
}