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

typedef struct {
  uint64_t q_addr;
  uint64_t k_addr;
  uint32_t dim;
  uint32_t head_size;
  uint32_t kv_dim;
  uint32_t pos;
} rope_arg_t;

typedef struct {
  uint64_t x_addr;
  uint64_t size;
  uint32_t elements_per_thread;

  uint64_t global_core_reduce_addr;
  uint64_t global_reduce_addr;
} softmax_arg_t;

typedef struct {
  uint64_t a_addr;
  uint64_t b_addr;
  uint32_t size;
} accum_arg_t;

typedef struct {
  uint64_t hb_addr;
  uint64_t hb2_addr;
  uint32_t hidden_dim;
} swiglu_arg_t;

typedef struct {
  uint64_t sxb_addr;
  uint64_t sq_addr;
  uint64_t satt_addr;
  uint64_t key_cache_addr;
  uint64_t value_cache_addr;

  uint32_t n_heads;
  uint32_t seq_len;
  uint32_t head_size;
  uint32_t kv_dim;
  uint32_t kv_mul;
  uint32_t pos;
  uint32_t loff;
} attention_arg_t;
