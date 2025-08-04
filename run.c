/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/mman.h>

#include <stdio.h>
#include "vortex.h"
#include "kernels/kernels.h"
#include "perf.h"

// Globals for comparison mode
static bool g_enable_comparison = false;
// 用于启用性能分析的全局标志
static bool g_enable_timing_analysis = false;

typedef uint64_t vx_addr_h;

// ----------------------------------------------------------------------------
// Vortex
#define RT_CHECK(_expr)                                                        \
  do {                                                                         \
    int _ret = _expr;                                                          \
    if (0 == _ret)                                                             \
      break;                                                                   \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);                   \
    cleanup();                                                                 \
    exit(-1);                                                                  \
  } while (false)

static vx_device_h device = NULL;

static void cleanup() {
  if (device) {
    RT_CHECK(vx_dev_close(device));
    device = NULL;
  }
}

static const char *vx_gemv_kernel = "./kernels/build/gemv.vxbin";
static vx_buffer_h vx_gemv_buf = NULL;

static const char *vx_rmsnorm_kernel = "./kernels/build/rmsnorm.vxbin";
static vx_buffer_h vx_rmsnorm_buf = NULL;

static const char *vx_rope_kernel = "./kernels/build/rope.vxbin";
static vx_buffer_h vx_rope_buf = NULL;

// static const char *vx_softmax_kernel = "./kernels/build/softmax.vxbin";
// static vx_buffer_h vx_softmax_buf = NULL;

static const char *vx_accum_kernel = "./kernels/build/accum.vxbin";
static vx_buffer_h vx_accum_buf = NULL;

static const char *vx_swiglu_kernel = "./kernels/build/swiglu.vxbin";
static vx_buffer_h vx_swiglu_buf = NULL;

static const char *vx_multihead_attention_kernel = "./kernels/build/attention.vxbin";
static vx_buffer_h vx_attention_buf = NULL;

// static const char * vx_tcu_matmul_kernel = "./kernels/build/tcu_matmul.vxbin";
// static vx_buffer_h vx_tcu_matmul_buf = NULL;

__attribute__((constructor)) static void vortex_module_ctor() {
  // Open Vortex device connection
  RT_CHECK(vx_dev_open(&device));
}

__attribute__((destructor)) static void vortex_module_dtor() { cleanup(); }

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;

    // TODO: refactor this to use a more generic way of handling vortex weights
    vx_addr_h token_embedding_table_vx;
    vx_addr_h rms_att_weight_vx;
    vx_addr_h wq_vx;
    vx_addr_h wk_vx;
    vx_addr_h wv_vx;
    vx_addr_h wo_vx;
    vx_addr_h rms_ffn_weight_vx;
    vx_addr_h w1_vx;
    vx_addr_h w2_vx;
    vx_addr_h w3_vx;
    vx_addr_h rms_final_weight_vx;
    vx_addr_h wcls_vx;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)


    // For vortex
    vx_buffer_h x_vx_buf;
    vx_addr_h x_vx;

    vx_buffer_h xb_vx_buf;
    vx_addr_h xb_vx;

    vx_buffer_h xb2_vx_buf;
    vx_addr_h xb2_vx;

    vx_buffer_h hb_vx_buf;
    vx_addr_h hb_vx;

    vx_buffer_h hb2_vx_buf;
    vx_addr_h hb2_vx;

    vx_buffer_h q_vx_buf;
    vx_addr_h q_vx;

    vx_addr_h k_vx;
    vx_addr_h v_vx;

    vx_buffer_h att_vx_buf;
    vx_addr_h att_vx;

    vx_buffer_h logits_vx_buf;
    vx_addr_h logits_vx;

    vx_buffer_h key_cache_vx_buf;
    vx_addr_h key_cache_vx;

    vx_buffer_h value_cache_vx_buf;
    vx_addr_h value_cache_vx;
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;


//=============================== helper function definitions =========================

void compare_results(const char* name, const float* cpu_out, const float* gpu_out, int size) {
    int mismatches = 0;
    float max_diff = 0.0f;
    const float epsilon = 1e-3f; // Tolerance for float comparison

    for (int i = 0; i < size; ++i) {
        float diff = fabsf(cpu_out[i] - gpu_out[i]);
        if (diff > epsilon) {
            if (mismatches < 5) { // Print first 5 mismatches
                fprintf(stderr, "\033[31mMismatch in %s at index %d: CPU=%.6f, GPU=%.6f, Diff=%.6f\033[0m\n", name, i, cpu_out[i], gpu_out[i], diff);
            }
            mismatches++;
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }

    if (mismatches > 0) {
        fprintf(stderr, "\033[31m-------> %s: FAILED! Total mismatches: %d / %d. Max difference: %.6f\033[0m\n", name, mismatches, size, max_diff);
    } else {
        printf("\033[32m-------> %s: OK\033[0m\n", name);
    }
}


//====================================================================================

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }

#define MALLOC_VORTEX_STATE(name, size)                                        \
    RT_CHECK(                                                                  \
        vx_mem_alloc(device, size, VX_MEM_READ_WRITE, &s->name##_vx_buf));     \
    RT_CHECK(vx_mem_address(s->name##_vx_buf, &s->name##_vx));

    MALLOC_VORTEX_STATE(x, p->dim * sizeof(float));
    MALLOC_VORTEX_STATE(xb, p->dim * sizeof(float));
    MALLOC_VORTEX_STATE(xb2, p->dim * sizeof(float));
    MALLOC_VORTEX_STATE(hb, p->hidden_dim * sizeof(float));
    MALLOC_VORTEX_STATE(hb2, p->hidden_dim * sizeof(float));
    MALLOC_VORTEX_STATE(q, p->dim * sizeof(float));
    MALLOC_VORTEX_STATE(key_cache,
                        p->n_layers * p->seq_len * kv_dim * sizeof(float));
    MALLOC_VORTEX_STATE(value_cache,
                        p->n_layers * p->seq_len * kv_dim * sizeof(float));
    MALLOC_VORTEX_STATE(att, p->n_heads * p->seq_len * sizeof(float));
    MALLOC_VORTEX_STATE(logits, p->vocab_size * sizeof(float));

#undef MALLOC_VORTEX_STATE
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);

    RT_CHECK(vx_mem_free(s->x_vx_buf));
    RT_CHECK(vx_mem_free(s->xb_vx_buf));
    RT_CHECK(vx_mem_free(s->xb2_vx_buf));
    RT_CHECK(vx_mem_free(s->hb_vx_buf));
    RT_CHECK(vx_mem_free(s->hb2_vx_buf));
    RT_CHECK(vx_mem_free(s->q_vx_buf));
    RT_CHECK(vx_mem_free(s->att_vx_buf));
    RT_CHECK(vx_mem_free(s->logits_vx_buf));
    RT_CHECK(vx_mem_free(s->key_cache_vx_buf));
    RT_CHECK(vx_mem_free(s->value_cache_vx_buf));
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void memory_map_weights_vx(TransformerWeights *w, Config *p, vx_addr_h ptr,
                           int shared_weights) {
  (void)shared_weights;
  int head_size = p->dim / p->n_heads;

  size_t n_layers = p->n_layers;
  w->token_embedding_table_vx = ptr;

  ptr += p->vocab_size * p->dim * sizeof(float);
  w->rms_att_weight_vx = ptr;

  ptr += n_layers * p->dim * sizeof(float);
  w->wq_vx = ptr;

  ptr += n_layers * p->dim * (p->n_heads * head_size) * sizeof(float);
  w->wk_vx = ptr;

  ptr += n_layers * p->dim * (p->n_kv_heads * head_size) * sizeof(float);
  w->wv_vx = ptr;

  ptr += n_layers * p->dim * (p->n_kv_heads * head_size) * sizeof(float);
  w->wo_vx = ptr;

  ptr += n_layers * (p->n_heads * head_size) * p->dim * sizeof(float);
  w->rms_ffn_weight_vx = ptr;

  ptr += n_layers * p->dim * sizeof(float);
  w->w1_vx = ptr;

  ptr += n_layers * p->dim * p->hidden_dim * sizeof(float);
  w->w2_vx = ptr;

  ptr += n_layers * p->hidden_dim * p->dim * sizeof(float);
  w->w3_vx = ptr;

  ptr += n_layers * p->dim * p->hidden_dim * sizeof(float);
  w->rms_final_weight_vx = ptr;

  ptr += p->dim * sizeof(float);
  ptr += p->seq_len * head_size / 2 *
         sizeof(float); // skip what used to be freq_cis_real (for RoPE)
  ptr += p->seq_len * head_size / 2 *
         sizeof(float); // skip what used to be freq_cis_imag (for RoPE)
  w->wcls_vx = shared_weights ? w->token_embedding_table_vx : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);

    // vortex memory map the Transformer weights
    size_t weights_size = *file_size - sizeof(Config);
    vx_buffer_h weights_buf = NULL;
    vx_addr_h weights_vx_ptr = 0;

    RT_CHECK(
        vx_mem_alloc(device, weights_size, VX_MEM_READ_WRITE, &weights_buf));
    RT_CHECK(vx_copy_to_dev(weights_buf, weights_ptr, 0, weights_size));
    RT_CHECK(vx_mem_address(weights_buf, &weights_vx_ptr));
    memory_map_weights_vx(weights, config, weights_vx_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

int divUp(int a, int b) { return (a - 1) / b + 1; }

void rmsnorm_vx(vx_addr_h o, vx_addr_h x, vx_addr_h weight, int size) {
  rmsnorm_arg_t args = {};

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));

  // Upload kernel arguments
  args.size = size;
  args.o_addr = o;
  args.x_addr = x;
  args.w_addr = weight;

  int total_threads = num_warps * num_threads;
  args.elements_per_thread = divUp(size, total_threads);

  vx_buffer_h rmsnorm_args_buffer;
  RT_CHECK(vx_upload_bytes(device, &args, sizeof(rmsnorm_arg_t),
                           &rmsnorm_args_buffer));

  // Upload kernel
  RT_CHECK(vx_upload_kernel_file(device, vx_rmsnorm_kernel, &vx_rmsnorm_buf));

  // Start the kernel
  RT_CHECK(vx_start(device, vx_rmsnorm_buf, rmsnorm_args_buffer));

  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));


  // Free the buffers
  RT_CHECK(vx_mem_free(rmsnorm_args_buffer));
  RT_CHECK(vx_mem_free(vx_rmsnorm_buf));
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}



void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void matmul_vx(vx_addr_h xout, vx_addr_h x, vx_addr_h w, int n, int d) {
  gemv_arg_t args = {};

  // Upload kernel arguments
  args.n = n;
  args.d = d;
  args.w_addr = w;
  args.x_addr = x;
  args.xout_addr = xout;

  vx_buffer_h gemv_args_buffer;
  RT_CHECK(
      vx_upload_bytes(device, &args, sizeof(gemv_arg_t), &gemv_args_buffer));

  // Upload kernel
  RT_CHECK(vx_upload_kernel_file(device, vx_gemv_kernel, &vx_gemv_buf));

  // Start the kernel
  RT_CHECK(vx_start(device, vx_gemv_buf, gemv_args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Free the buffers
  RT_CHECK(vx_mem_free(gemv_args_buffer));
  RT_CHECK(vx_mem_free(vx_gemv_buf));
}

void rope_encoding(int dim, int kv_dim, int head_size, float pos, float *q,
                   float *k) {
  for (int i = 0; i < dim; i += 2) {
    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
    for (int v = 0; v < rotn; v++) {
      float *vec = v == 0 ? q : k; // the vector to rotate (query or key)
      float v0 = vec[i];
      float v1 = vec[i + 1];
      vec[i] = v0 * fcr - v1 * fci;
      vec[i + 1] = v0 * fci + v1 * fcr;
    }
  }
}

void rope_encoding_vx(int dim, int kv_dim, int head_size, float pos,
                      vx_addr_h q, vx_addr_h k) {
  rope_arg_t args = {};

  // Upload kernel arguments
  args.dim = dim;
  args.head_size = head_size;
  args.kv_dim = kv_dim;
  args.pos = pos;
  args.q_addr = q;
  args.k_addr = k;

  vx_buffer_h rope_args_buffer;
  RT_CHECK(
      vx_upload_bytes(device, &args, sizeof(rope_arg_t), &rope_args_buffer));

  // Upload kernel
  RT_CHECK(vx_upload_kernel_file(device, vx_rope_kernel, &vx_rope_buf));

  // Start the kernel
  RT_CHECK(vx_start(device, vx_rope_buf, rope_args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  RT_CHECK(vx_mem_free(rope_args_buffer));
  RT_CHECK(vx_mem_free(vx_rope_buf));
}


void multihead_attention(float *sxb, float *sq, float *sk, float *sv,
                         float *satt, float *key_cache, float *value_cache,
                         int n_heads, int seq_len, int head_size, int kv_dim,
                         int kv_mul, int pos, int loff) {
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < n_heads; h++) {
    // get the query vector for this head
    float *q = sq + h * head_size;
    // attention scores for this head
    float *att = satt + h * seq_len;
    // iterate over all timesteps, including the current one
    for (int t = 0; t <= pos; t++) {
      // get the key vector for this head and at this timestep
      float *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
      // calculate the attention score as the dot product of q and k
      float score = 0.0f;
      for (int i = 0; i < head_size; i++) {
        score += q[i] * k[i];
      }
      score /= sqrtf(head_size);
      // save the score to the attention buffer
      att[t] = score;
    }

    // softmax the scores to get attention weights, from 0..pos inclusively
    softmax(att, pos + 1);

    // weighted sum of the values, store back into xb
    float *xb = sxb + h * head_size;
    memset(xb, 0, head_size * sizeof(float));
    for (int t = 0; t <= pos; t++) {
      // get the value vector for this head and at this timestep
      float *v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
      // get the attention weight for this timestep
      float a = att[t];
      // accumulate the weighted value into xb
      for (int i = 0; i < head_size; i++) {
        xb[i] += a * v[i];
      }
    }
  }
}

void multihead_attention_vx(vx_addr_h sxb, vx_addr_h sq, vx_addr_h sk,
                            vx_addr_h sv, vx_addr_h satt, vx_addr_h key_cache,
                            vx_addr_h value_cache, int n_heads, int seq_len,
                            int head_size, int kv_dim, int kv_mul, int pos,
                            int loff) {
  (void)sk;
  (void)sv;
  attention_arg_t args = {};

  // Upload kernel arguments
  args.n_heads = n_heads;
  args.seq_len = seq_len;
  args.head_size = head_size;
  args.kv_dim = kv_dim;
  args.kv_mul = kv_mul;
  args.pos = pos;
  args.loff = loff;
  args.sxb_addr = sxb;
  args.sq_addr = sq;
  args.satt_addr = satt;
  args.key_cache_addr = key_cache;
  args.value_cache_addr = value_cache;

  vx_buffer_h attention_args_buffer;
  RT_CHECK(vx_upload_bytes(device, &args, sizeof(attention_arg_t),
                           &attention_args_buffer));

  // Upload kernel
  RT_CHECK(vx_upload_kernel_file(device, vx_multihead_attention_kernel,
                                 &vx_attention_buf));
  // Start the kernel
  RT_CHECK(vx_start(device, vx_attention_buf, attention_args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Free the buffers
  RT_CHECK(vx_mem_free(attention_args_buffer));
  RT_CHECK(vx_mem_free(vx_attention_buf));
}

void accum(float *a, float *b, int size) {
  // accumulate b into a
  for (int i = 0; i < size; i++) {
    a[i] += b[i];
  }
}

void accum_vx(vx_addr_h a, vx_addr_h b, int size) {
  accum_arg_t args = {};

  // Upload kernel arguments
  args.size = size;
  args.a_addr = a;
  args.b_addr = b;

  vx_buffer_h accum_args_buffer;
  RT_CHECK(
      vx_upload_bytes(device, &args, sizeof(accum_arg_t), &accum_args_buffer));

  // Upload kernel
  RT_CHECK(vx_upload_kernel_file(device, vx_accum_kernel, &vx_accum_buf));

  // Start the kernel
  RT_CHECK(vx_start(device, vx_accum_buf, accum_args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  RT_CHECK(vx_mem_free(accum_args_buffer));
  RT_CHECK(vx_mem_free(vx_accum_buf));
}

void swiglu(float *hb, float *hb2, int hidden_dim) {
  for (int i = 0; i < hidden_dim; i++) {
    float val = hb[i];
    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    val *= (1.0f / (1.0f + expf(-val)));
    // elementwise multiply with w3(x)
    val *= hb2[i];
    hb[i] = val;
  }
}

void swiglu_vx(vx_addr_h hb, vx_addr_h hb2, int hidden_dim) {
  swiglu_arg_t args = {};

  // Upload kernel arguments
  args.hidden_dim = hidden_dim;
  args.hb_addr = hb;
  args.hb2_addr = hb2;

  vx_buffer_h swiglu_args_buffer;
  RT_CHECK(vx_upload_bytes(device, &args, sizeof(swiglu_arg_t),
                           &swiglu_args_buffer));

  // Upload kernel
  RT_CHECK(vx_upload_kernel_file(device, vx_swiglu_kernel, &vx_swiglu_buf));

  // Start the kernel
  RT_CHECK(vx_start(device, vx_swiglu_buf, swiglu_args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  RT_CHECK(vx_mem_free(swiglu_args_buffer));
  RT_CHECK(vx_mem_free(vx_swiglu_buf));
}

// float* forward(Transformer* transformer, int token, int pos) {

//     // a few convenience variables
//     Config* p = &transformer->config;
//     TransformerWeights* w = &transformer->weights;
//     RunState* s = &transformer->state;
//     float *x = s->x;
//     int dim = p->dim;
//     int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
//     int kv_mul = p->n_heads / p->n_kv_heads;
//     int hidden_dim =  p->hidden_dim;
//     int head_size = dim / p->n_heads;

//     // --- Performance Analysis Variables ---
//     PerfData pos_perf_data[NUM_OPS];
//     if (g_enable_timing_analysis) {
//         init_perf_data(pos_perf_data, NUM_OPS);
//     }
//     long long start_us, end_us;
//     char dim_str_buffer[32];
//     // --- End Performance Analysis Variables ---

//     // copy the token embedding into x
//     float *content_row = w->token_embedding_table + token * dim;

//     vx_buffer_h x_buf = NULL;
//     vx_addr_h x_vx = 0;
//     RT_CHECK(
//         vx_mem_alloc(device, dim * sizeof(float), VX_MEM_READ_WRITE, &x_buf));
//     RT_CHECK(vx_mem_address(x_buf, &x_vx));
//     RT_CHECK(vx_copy_to_dev(x_buf, content_row, 0, dim * sizeof(float)));


//     // forward all the layers
//     for(unsigned long long l = 0; l < p->n_layers; l++) {

//         // Attention rmsnorm
//         if (g_enable_comparison) {
//             float* cpu_out = malloc(dim * sizeof(float));
//             printf("\n[L%llu P%d] Comparing RMSNorm (Attention)...\n", l, pos);
//             rmsnorm(cpu_out, x, w->rms_att_weight + l*dim, dim);
//             printf("finished cpu rmsnorm\n");
//                  rmsnorm_vx(s->xb_vx, x_vx, w->rms_att_weight_vx + l * dim * sizeof(float),
//                  dim);
//             printf("finished gpu rmsnorm\n");
//             compare_results("rmsnorm_att", cpu_out, s->xb, dim);
//             free(cpu_out);
//         } else if (g_enable_timing_analysis) {
//             start_us = time_in_us();
//             rmsnorm_vx(s->xb_vx, x_vx, w->rms_att_weight_vx + l*dim*sizeof(float), dim);
//             end_us = time_in_us();
//             record_perf(pos_perf_data, OP_RMSNORM_ATT, start_us, end_us, NULL);
//         } else {
//             rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
//         }

//         // key and value point to the kv cache
//         int loff = l * p->seq_len * kv_dim;
//         s->k = s->key_cache + loff * sizeof(float) + pos * kv_dim * sizeof(float);
//         s->v = s->value_cache + loff * sizeof(float) + pos * kv_dim * sizeof(float);

//         // QKV matmuls for this position
//         if (g_enable_comparison) {
//             float* cpu_q = malloc(dim * sizeof(float));
//             float* cpu_k = malloc(kv_dim * sizeof(float));
//             float* cpu_v = malloc(kv_dim * sizeof(float));
//             printf("[L%llu P%d] Comparing MatMul (Q, K, V)...\n", l, pos);
//             matmul(cpu_q, s->xb, w->wq + l*dim*dim, dim, dim);
//             matmul_vx(s->q_vx, s->xb_vx, w->wq_vx + l*dim*dim*sizeof(float), dim, dim);
//             compare_results("matmul_q", cpu_q, s->q, dim);

//             matmul(cpu_k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
//             matmul_vx(s->k_vx, s->xb_vx, w->wk_vx + l*dim*kv_dim*sizeof(float), dim, kv_dim);
//             compare_results("matmul_k", cpu_k, s->k, kv_dim);

//             matmul(cpu_v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
//             matmul_vx(s->v_vx, s->xb_vx, w->wv_vx + l*dim*kv_dim*sizeof(float), dim, kv_dim);
//             compare_results("matmul_v", cpu_v, s->v, kv_dim);

//             free(cpu_q); free(cpu_k); free(cpu_v);
//         } else if (g_enable_timing_analysis) {
//             // Matmul Q
//             start_us = time_in_us();
//             matmul_vx(s->q_vx, s->xb_vx, w->wq_vx + l*dim*dim*sizeof(float), dim, dim);
//             end_us = time_in_us();
//             snprintf(dim_str_buffer, sizeof(dim_str_buffer), "[1, %d] x [%d, %d]", dim, dim, dim);
//             record_perf(pos_perf_data, OP_MATMUL_Q, start_us, end_us, dim_str_buffer);
            
//             // Matmul K
//             start_us = time_in_us();
//             matmul_vx(s->k_vx, s->xb_vx, w->wk_vx + l * dim * kv_dim * sizeof(float), dim, kv_dim);
//             end_us = time_in_us();
//             snprintf(dim_str_buffer, sizeof(dim_str_buffer), "[1, %d] x [%d, %d]", dim, dim, kv_dim);
//             record_perf(pos_perf_data, OP_MATMUL_K, start_us, end_us, dim_str_buffer);
            
//             // Matmul V
//             start_us = time_in_us();
//             matmul_vx(s->v_vx, s->xb_vx, w->wv_vx + l * dim * kv_dim * sizeof(float),dim, kv_dim);
//             end_us = time_in_us();
//             snprintf(dim_str_buffer, sizeof(dim_str_buffer), "[1, %d] x [%d, %d]", dim, dim, kv_dim);
//             record_perf(pos_perf_data, OP_MATMUL_V, start_us, end_us, dim_str_buffer);
//         } else {
//             matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
//             matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
//             matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
//         }

//         // RoPE relative positional encoding
//         if (g_enable_comparison) {
//             float* cpu_q = malloc(dim * sizeof(float));
//             float* cpu_k = malloc(kv_dim * sizeof(float));
//             memcpy(cpu_q, s->q, dim * sizeof(float));
//             memcpy(cpu_k, s->k, kv_dim * sizeof(float));
//             printf("[L%llu P%d] Comparing RoPE...\n", l, pos);
//             rope_encoding(dim, kv_dim, head_size, pos, cpu_q, cpu_k);
//             rope_encoding_vx(dim, kv_dim, head_size, pos, s->q_vx, s->k_vx);
//             compare_results("rope_q", cpu_q, s->q, dim);
//             compare_results("rope_k", cpu_k, s->k, kv_dim);
//             free(cpu_q); free(cpu_k);
//         } else if (g_enable_timing_analysis) {
//             start_us = time_in_us();
//             rope_encoding_vx(dim, kv_dim, head_size, pos, s->q_vx, s->k_vx);
//             end_us = time_in_us();
//             record_perf(pos_perf_data, OP_ROPE, start_us, end_us, NULL);
//         } else {
//             rope_encoding(dim, kv_dim, head_size, pos, s->q, s->k);
//         }

//         // Multihead attention
//         if (g_enable_comparison) {
//             float* cpu_xb = malloc(dim * sizeof(float));
//             float* cpu_att = malloc(p->n_heads * p->seq_len * sizeof(float));
//             memcpy(cpu_xb, s->xb, dim * sizeof(float));
//             memcpy(cpu_att, s->att, p->n_heads * p->seq_len * sizeof(float));
//             printf("[L%llu P%d] Comparing Multi-Head Attention...\n", l, pos);
//             multihead_attention(cpu_xb, s->q, s->k, s->v, cpu_att, s->key_cache, s->value_cache, p->n_heads, p->seq_len, head_size, kv_dim, kv_mul, pos, loff);
//             multihead_attention_vx(s->xb_vx, s->q_vx, s->k_vx, s->v_vx, s->att_vx,
//                              s->key_cache_vx, s->value_cache_vx, p->n_heads,
//                              p->seq_len, head_size, kv_dim, kv_mul, pos, loff);
//             compare_results("multihead_attention_xb", cpu_xb, s->xb, dim);
//             free(cpu_xb);
//             free(cpu_att);
//         } else if (g_enable_timing_analysis) {
//             start_us = time_in_us();
//             multihead_attention_vx(s->xb_vx, s->q_vx, s->k_vx, s->v_vx, s->att_vx,
//                              s->key_cache_vx, s->value_cache_vx, p->n_heads,
//                              p->seq_len, head_size, kv_dim, kv_mul, pos, loff);
//             end_us = time_in_us();
//             record_perf(pos_perf_data, OP_ATTENTION, start_us, end_us, NULL);
//         } else {
//             multihead_attention(s->xb, s->q, s->k, s->v, s->att, s->key_cache, s->value_cache, p->n_heads, p->seq_len, head_size, kv_dim, kv_mul, pos, loff);
//         }

//         // Final matmul (WO)
//         if (g_enable_comparison) {
//             float* cpu_out = malloc(dim * sizeof(float));
//             printf("[L%llu P%d] Comparing MatMul (WO)...\n", l, pos);
//             matmul(cpu_out, s->xb, w->wo + l*dim*dim, dim, dim);
//             matmul_vx(s->xb2_vx, s->xb_vx, w->wo_vx + l * dim * dim * sizeof(float),
//                 dim, dim);
//             compare_results("matmul_wo", cpu_out, s->xb2, dim);
//             free(cpu_out);
//         } else if (g_enable_timing_analysis) {
//             start_us = time_in_us();
//             matmul_vx(s->xb2_vx, s->xb_vx, w->wo_vx + l * dim * dim * sizeof(float),
//                 dim, dim);
//             end_us = time_in_us();
//             snprintf(dim_str_buffer, sizeof(dim_str_buffer), "[1, %d] x [%d, %d]", dim, dim, dim);
//             record_perf(pos_perf_data, OP_MATMUL_WO, start_us, end_us, dim_str_buffer);
//         } else {
//             matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);
//         }

//         // Residual connection
//         if (g_enable_comparison) {
//             float* cpu_x = malloc(dim * sizeof(float));
//             memcpy(cpu_x, x, dim * sizeof(float));
//             printf("[L%llu P%d] Comparing Accum (Attention)...\n", l, pos);
//             accum(cpu_x, s->xb2, dim);
//             accum_vx(x_vx, s->xb2_vx, dim);
//             compare_results("accum_att", cpu_x, x, dim);
//             free(cpu_x);
//         } else if (g_enable_timing_analysis) {
//             start_us = time_in_us();
//             accum_vx(x_vx, s->xb2_vx, dim);
//             end_us = time_in_us();
//             record_perf(pos_perf_data, OP_ACCUM_ATT, start_us, end_us, NULL);
//         } else {
//             accum(x, s->xb2, dim);
//         }

//         // FFN rmsnorm
//         if (g_enable_comparison) {
//             float* cpu_out = malloc(dim * sizeof(float));
//             printf("[L%llu P%d] Comparing RMSNorm (FFN)...\n", l, pos);
//             rmsnorm(cpu_out, x, w->rms_ffn_weight + l*dim, dim);
//             rmsnorm_vx(s->xb_vx, x_vx, w->rms_ffn_weight_vx + l * dim * sizeof(float),
//                  dim);
//             compare_results("rmsnorm_ffn", cpu_out, s->xb, dim);
//             free(cpu_out);
//         } else if (g_enable_timing_analysis) {
//             start_us = time_in_us();
//             rmsnorm_vx(s->xb_vx, x_vx, w->rms_ffn_weight_vx + l * dim * sizeof(float),
//                  dim);
//             end_us = time_in_us();
//             record_perf(pos_perf_data, OP_RMSNORM_FFN, start_us, end_us, NULL);
//         } else {
//             rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
//         }
        
//         // FFN matmuls (w1, w3)
//         if (g_enable_comparison) {
//             float* cpu_hb = malloc(hidden_dim * sizeof(float));
//             float* cpu_hb2 = malloc(hidden_dim * sizeof(float));
//             printf("[L%llu P%d] Comparing MatMul (W1, W3)...\n", l, pos);
//             matmul(cpu_hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
//             matmul_vx(s->hb_vx, s->xb_vx,
//                 w->w1_vx + l * dim * hidden_dim * sizeof(float), dim,
//                 hidden_dim);
//             compare_results("matmul_w1", cpu_hb, s->hb, hidden_dim);
//             matmul(cpu_hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
//             matmul_vx(s->hb2_vx, s->xb_vx,
//                 w->w3_vx + l * dim * hidden_dim * sizeof(float), dim,
//                 hidden_dim);
//             compare_results("matmul_w3", cpu_hb2, s->hb2, hidden_dim);
//             free(cpu_hb); free(cpu_hb2);
//         } else if (g_enable_timing_analysis) {
//             // Matmul W1
//             start_us = time_in_us();
//             matmul_vx(s->hb_vx, s->xb_vx,
//                 w->w1_vx + l * dim * hidden_dim * sizeof(float), dim,
//                 hidden_dim);
//             end_us = time_in_us();
//             snprintf(dim_str_buffer, sizeof(dim_str_buffer), "[1, %d] x [%d, %d]", dim, dim, hidden_dim);
//             record_perf(pos_perf_data, OP_MATMUL_W1, start_us, end_us, dim_str_buffer);

//             // Matmul W3
//             start_us = time_in_us();
//             matmul_vx(s->hb2_vx, s->xb_vx,
//                 w->w3_vx + l * dim * hidden_dim * sizeof(float), dim,
//                 hidden_dim);
//             end_us = time_in_us();
//             snprintf(dim_str_buffer, sizeof(dim_str_buffer), "[1, %d] x [%d, %d]", dim, dim, hidden_dim);
//             record_perf(pos_perf_data, OP_MATMUL_W3, start_us, end_us, dim_str_buffer);
//         } else {
//             matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
//             matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
//         }

//         // SwiGLU
//         if (g_enable_comparison) {
//             float* cpu_hb = malloc(hidden_dim * sizeof(float));
//             memcpy(cpu_hb, s->hb, hidden_dim * sizeof(float));
//             printf("[L%llu P%d] Comparing SwiGLU...\n", l, pos);
//             swiglu(cpu_hb, s->hb2, hidden_dim);
//             swiglu_vx(s->hb_vx, s->hb2_vx, hidden_dim);
//             compare_results("swiglu", cpu_hb, s->hb, hidden_dim);
//             free(cpu_hb);
//         } else if (g_enable_timing_analysis) {
//             start_us = time_in_us();
//             swiglu_vx(s->hb_vx, s->hb2_vx, hidden_dim);
//             end_us = time_in_us();
//             record_perf(pos_perf_data, OP_SWIGLU, start_us, end_us, NULL);
//         } else {
//             swiglu(s->hb, s->hb2, hidden_dim);
//         }

//         // Final FFN matmul (W2)
//         if (g_enable_comparison) {
//             float* cpu_out = malloc(dim * sizeof(float));
//             printf("[L%llu P%d] Comparing MatMul (W2)...\n", l, pos);
//             matmul(cpu_out, s->hb, w->w2 + l*hidden_dim*dim, hidden_dim, dim);
//             matmul_vx(s->xb_vx, s->hb_vx,
//                 w->w2_vx + l * dim * hidden_dim * sizeof(float), hidden_dim,
//                 dim);
//             compare_results("matmul_w2", cpu_out, s->xb, dim);
//             free(cpu_out);
//         } else if (g_enable_timing_analysis) {
//             start_us = time_in_us();
//             matmul_vx(s->xb_vx, s->hb_vx,
//                 w->w2_vx + l * dim * hidden_dim * sizeof(float), hidden_dim,
//                 dim);
//             end_us = time_in_us();
//             snprintf(dim_str_buffer, sizeof(dim_str_buffer), "[1, %d] x [%d, %d]", hidden_dim, hidden_dim, dim);
//             record_perf(pos_perf_data, OP_MATMUL_W2, start_us, end_us, dim_str_buffer);
//         } else {
//             matmul(s->xb, s->hb, w->w2 + l*hidden_dim*dim, hidden_dim, dim);
//         }

//         // Residual connection
//         if (g_enable_comparison) {
//             float* cpu_x = malloc(dim * sizeof(float));
//             memcpy(cpu_x, x, dim * sizeof(float));
//             printf("[L%llu P%d] Comparing Accum (FFN)...\n", l, pos);
//             accum(cpu_x, s->xb, dim);
//             accum_vx(x_vx, s->xb_vx, dim);
//             compare_results("accum_ffn", cpu_x, x, dim);
//             free(cpu_x);
//         } else if (g_enable_timing_analysis) {
//             start_us = time_in_us();
//             accum_vx(x_vx, s->xb_vx, dim);
//             end_us = time_in_us();
//             record_perf(pos_perf_data, OP_ACCUM_FFN, start_us, end_us, NULL);
//         } else {
//             accum(x, s->xb, dim);
//         }
//     }

//     // Final rmsnorm
//     if (g_enable_comparison) {
//         float* cpu_out = malloc(dim * sizeof(float));
//         float* x_input_copy = malloc(dim * sizeof(float));
//         memcpy(x_input_copy, x, dim * sizeof(float));
//         printf("[P%d] Comparing RMSNorm (Final)...\n", pos);
//         rmsnorm(cpu_out, x_input_copy, w->rms_final_weight, dim);

//         rmsnorm_vx(x_vx, x_vx, w->rms_final_weight_vx, dim);

//         compare_results("rmsnorm_final", cpu_out, x, dim);
//         free(cpu_out);
//         free(x_input_copy);
//     } else if (g_enable_timing_analysis) {
//         start_us = time_in_us();
//         rmsnorm_vx(x_vx, x_vx, w->rms_final_weight_vx, dim);
//         end_us = time_in_us();
//         record_perf(pos_perf_data, OP_RMSNORM_FINAL, start_us, end_us, NULL);
//     } else {
//         rmsnorm(x, x, w->rms_final_weight, dim);
//     }

//     // Classifier into logits
//     if (g_enable_comparison) {
//         float* cpu_logits = malloc(p->vocab_size * sizeof(float));
//         printf("[P%d] Comparing MatMul (Classifier)...\n", pos);
//         matmul(cpu_logits, x, w->wcls, p->dim, p->vocab_size);

//         matmul_vx(s->logits_vx, x_vx, w->wcls_vx, p->dim, p->vocab_size);

//         compare_results("matmul_classifier", cpu_logits, s->logits, p->vocab_size);
//         free(cpu_logits);
//     } else if (g_enable_timing_analysis) {
//         start_us = time_in_us();

//         matmul_vx(s->logits_vx, x_vx, w->wcls_vx, p->dim, p->vocab_size);

//         end_us = time_in_us();
//         snprintf(dim_str_buffer, sizeof(dim_str_buffer), "[1, %d] x [%d, %d]", p->dim, p->dim, p->vocab_size);
//         record_perf(pos_perf_data, OP_MATMUL_CLS, start_us, end_us, dim_str_buffer);
//     } else {
//         matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
//     }

//     // --- Print Performance Analysis Report ---
//     if (g_enable_timing_analysis) {
//         long long total_pos_time_us = 0;
//         for (int i = 0; i < NUM_OPS; i++) {
//             total_pos_time_us += pos_perf_data[i].total_time_us;
//         }

//         printf("\n\033[1;33m--- Timing Analysis for Position %d ---\033[0m\n", pos);
//         printf("\033[1;33m----------------------------------------------------------------------------------------\033[0m\n");
//         printf("\033[1;33m%-24s | %-24s | %12s | %s\033[0m\n", "Operator", "Dimensions", "Time (us)", "Percentage");
//         printf("\033[1;33m----------------------------------------------------------------------------------------\033[0m\n");

//         for (int i = 0; i < NUM_OPS; i++) {
//             if (pos_perf_data[i].call_count > 0) {
//                 double percentage = (total_pos_time_us > 0) ? ((double)pos_perf_data[i].total_time_us / total_pos_time_us * 100.0) : 0.0;
//                 printf("%-24s | %-24s | %12lld | %9.2f%%\n",
//                        op_names[i],
//                        pos_perf_data[i].dim_info,
//                        pos_perf_data[i].total_time_us,
//                        percentage);
//             }
//         }
//         printf("----------------------------------------------------------------------------------------\n");
//         printf("\033[1;33m%-24s | %-24s | %12lld | %9.2f%%\033[0m\n", "Total Position Time", "", total_pos_time_us, 100.0);
//         printf("\033[1;33m----------------------------------------------------------------------------------------\033[0m\n\n");
//         fflush(stdout);
//     }
//     // --- End Report ---
//     RT_CHECK(vx_copy_from_dev(s->logits, s->logits_vx_buf, 0,
//                               p->vocab_size * sizeof(float)));
//     RT_CHECK(vx_mem_free(x_buf));
//     return s->logits;
// }

float* forward(Transformer* transformer, int token, int pos) {
    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // --- Performance Analysis Variables ---
    PerfData pos_perf_data[NUM_OPS];
    if (g_enable_timing_analysis) {
        init_perf_data(pos_perf_data, NUM_OPS);
    }
    long long start_us, end_us;
    char dim_str_buffer[32];
    // --- End Performance Analysis Variables ---

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    // If on a GPU path, upload the initial activation 'x' to the device
    if (g_enable_comparison || g_enable_timing_analysis) {
        RT_CHECK(vx_copy_to_dev(s->x_vx_buf, x, 0, dim * sizeof(float)));
    }

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        int loff = l * p->seq_len * kv_dim;

        // Attention rmsnorm
        if (g_enable_comparison) {
            float* cpu_out = malloc(dim * sizeof(float));
            float* gpu_out = malloc(dim * sizeof(float));
            printf("\n[L%llu P%d] Comparing RMSNorm (Attention)...\n", l, pos);

            // CPU path
            rmsnorm(cpu_out, x, w->rms_att_weight + l*dim, dim);

            // GPU path
            rmsnorm_vx(s->xb_vx, s->x_vx, w->rms_att_weight_vx + l*dim*sizeof(float), dim);
            RT_CHECK(vx_copy_from_dev(gpu_out, s->xb_vx_buf, 0, dim*sizeof(float)));

            compare_results("rmsnorm_att", cpu_out, gpu_out, dim);
            memcpy(s->xb, cpu_out, dim * sizeof(float)); // Update host state for next op

            free(cpu_out);
            free(gpu_out);
        } else if (g_enable_timing_analysis) {
            start_us = time_in_us();
            rmsnorm_vx(s->xb_vx, s->x_vx, w->rms_att_weight_vx + l*dim*sizeof(float), dim);
            end_us = time_in_us();
            record_perf(pos_perf_data, OP_RMSNORM_ATT, start_us, end_us, NULL);
        } else {
            rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        }

        // key and value pointers/addresses for the kv cache
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;
        vx_addr_h k_vx_pos = s->key_cache_vx + (loff + pos * kv_dim) * sizeof(float);
        vx_addr_h v_vx_pos = s->value_cache_vx + (loff + pos * kv_dim) * sizeof(float);

        // QKV matmuls for this position
        if (g_enable_comparison) {
            printf("[L%llu P%d] Comparing MatMul (Q, K, V)...\n", l, pos);
            // Matmul Q
            float* cpu_q = malloc(dim*sizeof(float)); float* gpu_q = malloc(dim*sizeof(float));
            matmul(cpu_q, s->xb, w->wq + l*dim*dim, dim, dim);
            matmul_vx(s->q_vx, s->xb_vx, w->wq_vx + l*dim*dim*sizeof(float), dim, dim);
            RT_CHECK(vx_copy_from_dev(gpu_q, s->q_vx_buf, 0, dim*sizeof(float)));
            compare_results("matmul_q", cpu_q, gpu_q, dim);
            memcpy(s->q, cpu_q, dim*sizeof(float));
            free(cpu_q); free(gpu_q);

            // Matmul K
            float* cpu_k = malloc(kv_dim*sizeof(float)); float* gpu_k = malloc(kv_dim*sizeof(float));
            matmul(cpu_k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
            matmul_vx(k_vx_pos, s->xb_vx, w->wk_vx + l*dim*kv_dim*sizeof(float), dim, kv_dim);
            RT_CHECK(vx_copy_from_dev(gpu_k, s->key_cache_vx_buf, (loff + pos * kv_dim) * sizeof(float), kv_dim*sizeof(float)));
            compare_results("matmul_k", cpu_k, gpu_k, kv_dim);
            memcpy(s->k, cpu_k, kv_dim*sizeof(float));
            free(cpu_k); free(gpu_k);

            // Matmul V
            float* cpu_v = malloc(kv_dim*sizeof(float)); float* gpu_v = malloc(kv_dim*sizeof(float));
            matmul(cpu_v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
            matmul_vx(v_vx_pos, s->xb_vx, w->wv_vx + l*dim*kv_dim*sizeof(float), dim, kv_dim);
            RT_CHECK(vx_copy_from_dev(gpu_v, s->value_cache_vx_buf, (loff + pos * kv_dim) * sizeof(float), kv_dim*sizeof(float)));
            compare_results("matmul_v", cpu_v, gpu_v, kv_dim);
            memcpy(s->v, cpu_v, kv_dim*sizeof(float));
            free(cpu_v); free(gpu_v);
        } else if (g_enable_timing_analysis) {
            start_us = time_in_us(); matmul_vx(s->q_vx, s->xb_vx, w->wq_vx + l*dim*dim*sizeof(float), dim, dim); end_us = time_in_us();
            snprintf(dim_str_buffer, sizeof(dim_str_buffer), "[1, %d] x [%d, %d]", dim, dim, dim); record_perf(pos_perf_data, OP_MATMUL_Q, start_us, end_us, dim_str_buffer);
            start_us = time_in_us(); matmul_vx(k_vx_pos, s->xb_vx, w->wk_vx + l*dim*kv_dim*sizeof(float), dim, kv_dim); end_us = time_in_us();
            snprintf(dim_str_buffer, sizeof(dim_str_buffer), "[1, %d] x [%d, %d]", dim, dim, kv_dim); record_perf(pos_perf_data, OP_MATMUL_K, start_us, end_us, dim_str_buffer);
            start_us = time_in_us(); matmul_vx(v_vx_pos, s->xb_vx, w->wv_vx + l*dim*kv_dim*sizeof(float), dim, kv_dim); end_us = time_in_us();
            snprintf(dim_str_buffer, sizeof(dim_str_buffer), "[1, %d] x [%d, %d]", dim, dim, kv_dim); record_perf(pos_perf_data, OP_MATMUL_V, start_us, end_us, dim_str_buffer);
        } else {
            matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
            matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
            matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
        }

        // RoPE relative positional encoding
        if (g_enable_comparison) {
            float* cpu_q = malloc(dim*sizeof(float)); memcpy(cpu_q, s->q, dim*sizeof(float));
            float* cpu_k = malloc(kv_dim*sizeof(float)); memcpy(cpu_k, s->k, kv_dim*sizeof(float));
            float* gpu_q = malloc(dim*sizeof(float)); float* gpu_k = malloc(kv_dim*sizeof(float));
            printf("[L%llu P%d] Comparing RoPE...\n", l, pos);

            rope_encoding(dim, kv_dim, head_size, pos, cpu_q, cpu_k);
            rope_encoding_vx(dim, kv_dim, head_size, pos, s->q_vx, k_vx_pos);

            RT_CHECK(vx_copy_from_dev(gpu_q, s->q_vx_buf, 0, dim*sizeof(float)));
            RT_CHECK(vx_copy_from_dev(gpu_k, s->key_cache_vx_buf, (loff + pos*kv_dim)*sizeof(float), kv_dim*sizeof(float)));
            
            compare_results("rope_q", cpu_q, gpu_q, dim);
            compare_results("rope_k", cpu_k, gpu_k, kv_dim);

            memcpy(s->q, cpu_q, dim*sizeof(float));
            memcpy(s->k, cpu_k, kv_dim*sizeof(float));

            free(cpu_q); free(cpu_k); free(gpu_q); free(gpu_k);
        } else if (g_enable_timing_analysis) {
            start_us = time_in_us();
            rope_encoding_vx(dim, kv_dim, head_size, pos, s->q_vx, k_vx_pos);
            end_us = time_in_us();
            record_perf(pos_perf_data, OP_ROPE, start_us, end_us, NULL);
        } else {
            rope_encoding(dim, kv_dim, head_size, pos, s->q, s->k);
        }

        // Multihead attention
        if (g_enable_comparison) {
            float* cpu_xb = malloc(dim * sizeof(float));
            float* gpu_xb = malloc(dim * sizeof(float));
            printf("[L%llu P%d] Comparing Multi-Head Attention...\n", l, pos);
            
            multihead_attention(cpu_xb, s->q, s->k, s->v, s->att, s->key_cache, s->value_cache, p->n_heads, p->seq_len, head_size, kv_dim, kv_mul, pos, loff);
            multihead_attention_vx(s->xb_vx, s->q_vx, 0, 0, s->att_vx, s->key_cache_vx, s->value_cache_vx, p->n_heads, p->seq_len, head_size, kv_dim, kv_mul, pos, loff);
            RT_CHECK(vx_copy_from_dev(gpu_xb, s->xb_vx_buf, 0, dim*sizeof(float)));
            
            compare_results("multihead_attention_xb", cpu_xb, gpu_xb, dim);
            memcpy(s->xb, cpu_xb, dim * sizeof(float));

            free(cpu_xb);
            free(gpu_xb);
        } else if (g_enable_timing_analysis) {
            start_us = time_in_us();
            multihead_attention_vx(s->xb_vx, s->q_vx, 0, 0, s->att_vx, s->key_cache_vx, s->value_cache_vx, p->n_heads, p->seq_len, head_size, kv_dim, kv_mul, pos, loff);
            end_us = time_in_us();
            record_perf(pos_perf_data, OP_ATTENTION, start_us, end_us, NULL);
        } else {
            multihead_attention(s->xb, s->q, s->k, s->v, s->att, s->key_cache, s->value_cache, p->n_heads, p->seq_len, head_size, kv_dim, kv_mul, pos, loff);
        }

        // Final matmul (WO)
        if (g_enable_comparison) {
            float* cpu_out = malloc(dim * sizeof(float));
            float* gpu_out = malloc(dim * sizeof(float));
            printf("[L%llu P%d] Comparing MatMul (WO)...\n", l, pos);

            matmul(cpu_out, s->xb, w->wo + l*dim*dim, dim, dim);
            matmul_vx(s->xb2_vx, s->xb_vx, w->wo_vx + l*dim*dim*sizeof(float), dim, dim);
            RT_CHECK(vx_copy_from_dev(gpu_out, s->xb2_vx_buf, 0, dim * sizeof(float)));

            compare_results("matmul_wo", cpu_out, gpu_out, dim);
            memcpy(s->xb2, cpu_out, dim * sizeof(float));

            free(cpu_out);
            free(gpu_out);
        } else if (g_enable_timing_analysis) {
            start_us = time_in_us();
            matmul_vx(s->xb2_vx, s->xb_vx, w->wo_vx + l*dim*dim*sizeof(float), dim, dim);
            end_us = time_in_us();
            snprintf(dim_str_buffer, sizeof(dim_str_buffer), "[1, %d] x [%d, %d]", dim, dim, dim);
            record_perf(pos_perf_data, OP_MATMUL_WO, start_us, end_us, dim_str_buffer);
        } else {
            matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);
        }

        // Residual connection
        if (g_enable_comparison) {
            float* cpu_x = malloc(dim * sizeof(float)); memcpy(cpu_x, x, dim * sizeof(float));
            float* gpu_x = malloc(dim * sizeof(float));
            printf("[L%llu P%d] Comparing Accum (Attention)...\n", l, pos);
            
            accum(cpu_x, s->xb2, dim);
            accum_vx(s->x_vx, s->xb2_vx, dim);
            RT_CHECK(vx_copy_from_dev(gpu_x, s->x_vx_buf, 0, dim * sizeof(float)));

            compare_results("accum_att", cpu_x, gpu_x, dim);
            memcpy(x, cpu_x, dim * sizeof(float));

            free(cpu_x); free(gpu_x);
        } else if (g_enable_timing_analysis) {
            start_us = time_in_us();
            accum_vx(s->x_vx, s->xb2_vx, dim);
            end_us = time_in_us();
            record_perf(pos_perf_data, OP_ACCUM_ATT, start_us, end_us, NULL);
        } else {
            accum(x, s->xb2, dim);
        }

        // FFN rmsnorm
        if (g_enable_comparison) {
            float* cpu_out = malloc(dim * sizeof(float));
            float* gpu_out = malloc(dim * sizeof(float));
            printf("[L%llu P%d] Comparing RMSNorm (FFN)...\n", l, pos);

            rmsnorm(cpu_out, x, w->rms_ffn_weight + l*dim, dim);
            rmsnorm_vx(s->xb_vx, s->x_vx, w->rms_ffn_weight_vx + l*dim*sizeof(float), dim);
            RT_CHECK(vx_copy_from_dev(gpu_out, s->xb_vx_buf, 0, dim * sizeof(float)));

            compare_results("rmsnorm_ffn", cpu_out, gpu_out, dim);
            memcpy(s->xb, cpu_out, dim * sizeof(float));

            free(cpu_out); free(gpu_out);
        } else if (g_enable_timing_analysis) {
            start_us = time_in_us();
            rmsnorm_vx(s->xb_vx, s->x_vx, w->rms_ffn_weight_vx + l*dim*sizeof(float), dim);
            end_us = time_in_us();
            record_perf(pos_perf_data, OP_RMSNORM_FFN, start_us, end_us, NULL);
        } else {
            rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
        }
        
        // FFN matmuls (w1, w3)
        if (g_enable_comparison) {
            printf("[L%llu P%d] Comparing MatMul (W1, W3)...\n", l, pos);
            // Matmul W1
            float* cpu_hb = malloc(hidden_dim * sizeof(float)); float* gpu_hb = malloc(hidden_dim * sizeof(float));
            matmul(cpu_hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
            matmul_vx(s->hb_vx, s->xb_vx, w->w1_vx + l*dim*hidden_dim*sizeof(float), dim, hidden_dim);
            RT_CHECK(vx_copy_from_dev(gpu_hb, s->hb_vx_buf, 0, hidden_dim * sizeof(float)));
            compare_results("matmul_w1", cpu_hb, gpu_hb, hidden_dim);
            memcpy(s->hb, cpu_hb, hidden_dim * sizeof(float));
            free(cpu_hb); free(gpu_hb);

            // Matmul W3
            float* cpu_hb2 = malloc(hidden_dim * sizeof(float)); float* gpu_hb2 = malloc(hidden_dim * sizeof(float));
            matmul(cpu_hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
            matmul_vx(s->hb2_vx, s->xb_vx, w->w3_vx + l*dim*hidden_dim*sizeof(float), dim, hidden_dim);
            RT_CHECK(vx_copy_from_dev(gpu_hb2, s->hb2_vx_buf, 0, hidden_dim * sizeof(float)));
            compare_results("matmul_w3", cpu_hb2, gpu_hb2, hidden_dim);
            memcpy(s->hb2, cpu_hb2, hidden_dim * sizeof(float));
            free(cpu_hb2); free(gpu_hb2);
        } else if (g_enable_timing_analysis) {
            start_us = time_in_us(); matmul_vx(s->hb_vx, s->xb_vx, w->w1_vx + l*dim*hidden_dim*sizeof(float), dim, hidden_dim); end_us = time_in_us();
            snprintf(dim_str_buffer, sizeof(dim_str_buffer), "[1, %d] x [%d, %d]", dim, dim, hidden_dim); record_perf(pos_perf_data, OP_MATMUL_W1, start_us, end_us, dim_str_buffer);
            start_us = time_in_us(); matmul_vx(s->hb2_vx, s->xb_vx, w->w3_vx + l*dim*hidden_dim*sizeof(float), dim, hidden_dim); end_us = time_in_us();
            snprintf(dim_str_buffer, sizeof(dim_str_buffer), "[1, %d] x [%d, %d]", dim, dim, hidden_dim); record_perf(pos_perf_data, OP_MATMUL_W3, start_us, end_us, dim_str_buffer);
        } else {
            matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
            matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        }

        // SwiGLU
        if (g_enable_comparison) {
            float* cpu_hb = malloc(hidden_dim * sizeof(float)); memcpy(cpu_hb, s->hb, hidden_dim * sizeof(float));
            float* gpu_hb = malloc(hidden_dim * sizeof(float));
            printf("[L%llu P%d] Comparing SwiGLU...\n", l, pos);

            swiglu(cpu_hb, s->hb2, hidden_dim);
            swiglu_vx(s->hb_vx, s->hb2_vx, hidden_dim);
            RT_CHECK(vx_copy_from_dev(gpu_hb, s->hb_vx_buf, 0, hidden_dim * sizeof(float)));

            compare_results("swiglu", cpu_hb, gpu_hb, hidden_dim);
            memcpy(s->hb, cpu_hb, hidden_dim * sizeof(float));
            
            free(cpu_hb); free(gpu_hb);
        } else if (g_enable_timing_analysis) {
            start_us = time_in_us();
            swiglu_vx(s->hb_vx, s->hb2_vx, hidden_dim);
            end_us = time_in_us();
            record_perf(pos_perf_data, OP_SWIGLU, start_us, end_us, NULL);
        } else {
            swiglu(s->hb, s->hb2, hidden_dim);
        }

        // Final FFN matmul (W2)
        if (g_enable_comparison) {
            float* cpu_out = malloc(dim * sizeof(float));
            float* gpu_out = malloc(dim * sizeof(float));
            printf("[L%llu P%d] Comparing MatMul (W2)...\n", l, pos);

            matmul(cpu_out, s->hb, w->w2 + l*hidden_dim*dim, hidden_dim, dim);
            matmul_vx(s->xb_vx, s->hb_vx, w->w2_vx + l*hidden_dim*dim*sizeof(float), hidden_dim, dim);
            RT_CHECK(vx_copy_from_dev(gpu_out, s->xb_vx_buf, 0, dim * sizeof(float)));

            compare_results("matmul_w2", cpu_out, gpu_out, dim);
            memcpy(s->xb, cpu_out, dim * sizeof(float));

            free(cpu_out); free(gpu_out);
        } else if (g_enable_timing_analysis) {
            start_us = time_in_us();
            matmul_vx(s->xb_vx, s->hb_vx, w->w2_vx + l*hidden_dim*dim*sizeof(float), hidden_dim, dim);
            end_us = time_in_us();
            snprintf(dim_str_buffer, sizeof(dim_str_buffer), "[1, %d] x [%d, %d]", hidden_dim, hidden_dim, dim);
            record_perf(pos_perf_data, OP_MATMUL_W2, start_us, end_us, dim_str_buffer);
        } else {
            matmul(s->xb, s->hb, w->w2 + l*hidden_dim*dim, hidden_dim, dim);
        }

        // Residual connection
        if (g_enable_comparison) {
            float* cpu_x = malloc(dim * sizeof(float)); memcpy(cpu_x, x, dim * sizeof(float));
            float* gpu_x = malloc(dim * sizeof(float));
            printf("[L%llu P%d] Comparing Accum (FFN)...\n", l, pos);

            accum(cpu_x, s->xb, dim);
            accum_vx(s->x_vx, s->xb_vx, dim);
            RT_CHECK(vx_copy_from_dev(gpu_x, s->x_vx_buf, 0, dim * sizeof(float)));
            
            compare_results("accum_ffn", cpu_x, gpu_x, dim);
            memcpy(x, cpu_x, dim * sizeof(float));

            free(cpu_x); free(gpu_x);
        } else if (g_enable_timing_analysis) {
            start_us = time_in_us();
            accum_vx(s->x_vx, s->xb_vx, dim);
            end_us = time_in_us();
            record_perf(pos_perf_data, OP_ACCUM_FFN, start_us, end_us, NULL);
        } else {
            accum(x, s->xb, dim);
        }
    }

    // Final rmsnorm
    if (g_enable_comparison) {
        float* cpu_out = malloc(dim * sizeof(float)); memcpy(cpu_out, x, dim * sizeof(float));
        float* gpu_out = malloc(dim * sizeof(float));
        printf("[P%d] Comparing RMSNorm (Final)...\n", pos);

        rmsnorm(cpu_out, cpu_out, w->rms_final_weight, dim); // In-place on cpu_out
        rmsnorm_vx(s->x_vx, s->x_vx, w->rms_final_weight_vx, dim);
        RT_CHECK(vx_copy_from_dev(gpu_out, s->x_vx_buf, 0, dim * sizeof(float)));

        compare_results("rmsnorm_final", cpu_out, gpu_out, dim);
        memcpy(x, cpu_out, dim * sizeof(float));

        free(cpu_out); free(gpu_out);
    } else if (g_enable_timing_analysis) {
        start_us = time_in_us();
        rmsnorm_vx(s->x_vx, s->x_vx, w->rms_final_weight_vx, dim);
        end_us = time_in_us();
        record_perf(pos_perf_data, OP_RMSNORM_FINAL, start_us, end_us, NULL);
    } else {
        rmsnorm(x, x, w->rms_final_weight, dim);
    }

    // Classifier into logits
    if (g_enable_comparison) {
        float* cpu_logits = malloc(p->vocab_size * sizeof(float));
        float* gpu_logits = malloc(p->vocab_size * sizeof(float));
        printf("[P%d] Comparing MatMul (Classifier)...\n", pos);

        matmul(cpu_logits, x, w->wcls, p->dim, p->vocab_size);
        matmul_vx(s->logits_vx, s->x_vx, w->wcls_vx, p->dim, p->vocab_size);
        RT_CHECK(vx_copy_from_dev(gpu_logits, s->logits_vx_buf, 0, p->vocab_size * sizeof(float)));

        compare_results("matmul_classifier", cpu_logits, gpu_logits, p->vocab_size);
        memcpy(s->logits, cpu_logits, p->vocab_size * sizeof(float));

        free(cpu_logits); free(gpu_logits);
    } else if (g_enable_timing_analysis) {
        start_us = time_in_us();
        matmul_vx(s->logits_vx, s->x_vx, w->wcls_vx, p->dim, p->vocab_size);
        end_us = time_in_us();
        snprintf(dim_str_buffer, sizeof(dim_str_buffer), "[1, %d] x [%d, %d]", p->dim, p->dim, p->vocab_size);
        record_perf(pos_perf_data, OP_MATMUL_CLS, start_us, end_us, dim_str_buffer);
    } else {
        matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    }

    // For GPU paths, download the final logits to the host buffer
    if (g_enable_timing_analysis) {
        RT_CHECK(vx_copy_from_dev(s->logits, s->logits_vx_buf, 0, p->vocab_size * sizeof(float)));
    }

    // --- Print Performance Analysis Report ---
    if (g_enable_timing_analysis) {
        long long total_pos_time_us = 0;
        for (int i = 0; i < NUM_OPS; i++) {
            total_pos_time_us += pos_perf_data[i].total_time_us;
        }

        printf("\n\033[1;33m--- Timing Analysis for Position %d ---\033[0m\n", pos);
        printf("\033[1;33m----------------------------------------------------------------------------------------\033[0m\n");
        printf("\033[1;33m%-24s | %-24s | %12s | %s\033[0m\n", "Operator", "Dimensions", "Time (us)", "Percentage");
        printf("\033[1;33m----------------------------------------------------------------------------------------\033[0m\n");

        for (int i = 0; i < NUM_OPS; i++) {
            if (pos_perf_data[i].call_count > 0) {
                double percentage = (total_pos_time_us > 0) ? ((double)pos_perf_data[i].total_time_us / total_pos_time_us * 100.0) : 0.0;
                printf("%-24s | %-24s | %12lld | %9.2f%%\n",
                       op_names[i],
                       pos_perf_data[i].dim_info,
                       pos_perf_data[i].total_time_us,
                       percentage);
            }
        }
        printf("----------------------------------------------------------------------------------------\n");
        printf("\033[1;33m%-24s | %-24s | %12lld | %9.2f%%\033[0m\n", "Total Position Time", "", total_pos_time_us, 100.0);
        printf("\033[1;33m----------------------------------------------------------------------------------------\033[0m\n\n");
        fflush(stdout);
    }
    // --- End Report ---

    return s->logits;
}


// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    
    // 如果启用了对比模式，让正常输出更醒目
    if (g_enable_comparison) {
        // 使用绿色粗体显示正常文本输出
        printf("\033[1;32m%s\033[0m", piece);
    } else {
        printf("%s", piece);
    }
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // 如果启用对比模式，添加醒目的开始标识
    if (g_enable_comparison) {
        printf("\n\033[1;36m================================================================================\033[0m\n");
        printf("\033[1;36m🚀 开始文本生成 (对比模式已启用)\033[0m\n");
        printf("\033[1;36m💡 绿色粗体文字为AI生成的内容\033[0m\n");
        printf("\033[1;36m================================================================================\033[0m\n\n");
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // 现在会根据对比模式显示醒目的输出
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
        //TODO: 这里用于对比，仅打印1个pos的生成结果，所以直接退出
        break;
    }
    
    // 如果启用对比模式，添加醒目的结束标识
    if (g_enable_comparison) {
        printf("\n\n\033[1;36m================================================================================\033[0m\n");
        printf("\033[1;36m✅ 文本生成完成\033[0m\n");
        printf("\033[1;36m================================================================================\033[0m\n");
    } else {
        printf("\n");
    }

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // 如果启用对比模式，添加聊天模式的醒目标识
    if (g_enable_comparison) {
        printf("\n\033[1;35m================================================================================\033[0m\n");
        printf("\033[1;35m💬 聊天模式 (对比模式已启用)\033[0m\n");
        printf("\033[1;35m💡 Assistant回复将以绿色粗体显示\033[0m\n");
        printf("\033[1;35m================================================================================\033[0m\n\n");
    }

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            
            // 为Assistant回复添加醒目标识
            if (g_enable_comparison) {
                printf("\033[1;34mAssistant: \033[0m");
            } else {
                printf("Assistant: ");
            }
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // 现在会根据对比模式显示醒目的输出
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    fprintf(stderr, "  -c          enable comparison between CPU and Vortex GPGPU results\n");
    fprintf(stderr, "  -a          enable timing analysis for Vortex GPGPU operators\n"); 
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

        // Argument parsing
    if (argc < 2) {
        error_usage();
    }

    checkpoint_path = argv[1];

   
    for (int i = 2; i < argc; i++) {
        if (argv[i][0] != '-') { error_usage(); }
        
        char flag = argv[i][1];
        
        // --- 开始：修改命令行解析逻辑 ---
        if (flag == 'c') {
            g_enable_comparison = true;
            g_enable_timing_analysis = false; // 与分析模式互斥
            continue; // 无需参数
        }
        if (flag == 'a') { // 新增的分析模式标志
            g_enable_timing_analysis = true;
            g_enable_comparison = false; // 与对比模式互斥
            printf("Timing analysis mode enabled. Will run on Vortex GPGPU.\n");
            continue; // 无需参数
        }
        // --- 结束：修改命令行解析逻辑 ---

        // For flags with arguments, check if argument exists
        if (i + 1 >= argc) { error_usage(); }
        
        char *arg = argv[i + 1];
        
        if (flag == 't') { temperature = atof(arg); }
        else if (flag == 'p') { topp = atof(arg); }
        else if (flag == 's') { rng_seed = atoi(arg); }
        else if (flag == 'n') { steps = atoi(arg); }
        else if (flag == 'i') { prompt = arg; }
        else if (flag == 'z') { tokenizer_path = arg; }
        else if (flag == 'm') { mode = arg; }
        else if (flag == 'y') { system_prompt = arg; }
        else { error_usage(); }
        
        i++; // Increment i to skip the argument
    }


    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
