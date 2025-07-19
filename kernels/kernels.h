#pragma once

#include <stdint.h>

typedef struct {
  uint64_t xout_addr;
  uint64_t x_addr;
  uint64_t w_addr;
  uint32_t n;
  uint32_t d;
} gemv_arg_t;