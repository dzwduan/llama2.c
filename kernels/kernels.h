#pragma once

#include <stdint.h>

typedef struct {
  uint64_t xout_addr;
  uint64_t x_addr;
  uint64_t w_addr;
  uint32_t n;
  uint32_t d;
} gemv_arg_t;

typedef struct {
  uint64_t o_addr;
  uint64_t x_addr;
  uint64_t w_addr;
  uint64_t size;
  uint64_t elements_per_thread;

  uint64_t global_ss_addr;
  uint64_t global_sums_addr;
} rmsnorm_arg_t;
