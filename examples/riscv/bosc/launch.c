/* One-workgroup TileLang GEMM launch and self-check, linked into NR firmware.
 */
#include "kernel_config.h"
#include "nr_runtime.h"
#include <stddef.h>
#include <stdint.h>

/* MLIR ranked memref C ABI: allocated, aligned, offset, sizes, strides. */
typedef struct {
  int8_t *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} I8MemRef2D;
typedef struct {
  int32_t *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} I32MemRef2D;
_Static_assert(sizeof(void *) == 8, "The target ABI must be RV64");
_Static_assert(sizeof(I8MemRef2D) == 56, "MLIR rank-2 descriptor ABI mismatch");
_Static_assert(offsetof(I8MemRef2D, strides) == 40, "MLIR stride ABI mismatch");
_Static_assert(sizeof(I32MemRef2D) == 56, "MLIR accumulator ABI mismatch");

extern void _mlir_ciface_bosc_gemm(I8MemRef2D *, I8MemRef2D *, I32MemRef2D *);

/* Separate objects, 64-byte alignment and a 64-byte prefix satisfy
 * contract.abi. */
static struct {
  uint8_t before[64];
  int8_t data[BOSC_M * BOSC_K];
  uint8_t after[64];
} input_a __attribute__((aligned(64)));
static struct {
  uint8_t before[64];
  int8_t data[BOSC_N * BOSC_K];
  uint8_t after[64];
} input_b __attribute__((aligned(64)));
static struct {
  uint8_t before[64];
  int32_t data[BOSC_M * BOSC_N];
  uint8_t after[64];
} output_c __attribute__((aligned(64)));
static int32_t expected[BOSC_M * BOSC_N];

static int8_t a_value(int index, int test) {
  if (test == 2)
    return index % 3 == 0 ? -128 : (index % 3 == 1 ? 127 : 0);
  if (test == 3)
    return 0;
  return (int8_t)((3 * index + 1 + test) % 7 - 3);
}
static int8_t b_value(int index, int test) {
  if (test == 2)
    return (int8_t)((index % 2) ? 1 : -1);
  return (int8_t)((5 * index + 2 + test) % 9 - 4);
}
static int32_t c_value(int index, int test) {
  return test == 0 ? 0 : index % 5 - 2;
}
static void initialize(int test) {
  for (int i = 0; i < 64; ++i) {
    input_a.before[i] = input_a.after[i] = 0xa5;
    input_b.before[i] = input_b.after[i] = 0xa5;
    output_c.before[i] = output_c.after[i] = 0xa5;
  }
  for (int i = 0; i < BOSC_M * BOSC_K; ++i)
    input_a.data[i] = a_value(i, test);
  for (int i = 0; i < BOSC_N * BOSC_K; ++i)
    input_b.data[i] = b_value(i, test);
  for (int i = 0; i < BOSC_M * BOSC_N; ++i)
    output_c.data[i] = expected[i] = c_value(i, test);
  /* Scalar oracle stays in the image; C compilation disables vectorization. */
  for (int m = 0; m < BOSC_M; ++m)
    for (int n = 0; n < BOSC_N; ++n)
      for (int k = 0; k < BOSC_K; ++k)
        expected[m * BOSC_N + n] += (int32_t)input_a.data[m * BOSC_K + k] *
                                    (int32_t)input_b.data[n * BOSC_K + k];
}
static unsigned check(int test) {
  unsigned errors = 0;
  for (int i = 0; i < BOSC_M * BOSC_N; ++i)
    errors += output_c.data[i] != expected[i];
  for (int i = 0; i < BOSC_M * BOSC_K; ++i)
    errors += input_a.data[i] != a_value(i, test);
  for (int i = 0; i < BOSC_N * BOSC_K; ++i)
    errors += input_b.data[i] != b_value(i, test);
  for (int i = 0; i < 64; ++i) {
    errors += input_a.before[i] != 0xa5 || input_a.after[i] != 0xa5;
    errors += input_b.before[i] != 0xa5 || input_b.after[i] != 0xa5;
    errors += output_c.before[i] != 0xa5 || output_c.after[i] != 0xa5;
  }
  return errors;
}

int launch(void) {
  nr_puts("[tilelang-bosc] INT8 GEMM launch; grid=1x1x1\r\n");
  if (((uintptr_t)input_a.data | (uintptr_t)input_b.data |
       (uintptr_t)output_c.data) &
      63u) {
    nr_puts("[tilelang-bosc] entry alignment: FAIL\r\n");
    return 1;
  }
  I8MemRef2D a = {input_a.data, input_a.data, 0, {BOSC_M, BOSC_K}, {BOSC_K, 1}};
  I8MemRef2D b = {input_b.data, input_b.data, 0, {BOSC_N, BOSC_K}, {BOSC_K, 1}};
  I32MemRef2D c = {
      output_c.data, output_c.data, 0, {BOSC_M, BOSC_N}, {BOSC_N, 1}};
  unsigned total_errors = 0;
  for (int test = 0; test < 4; ++test) {
    initialize(test);
    ame_fence();
    _mlir_ciface_bosc_gemm(&a, &b, &c);
    ame_fence();
    unsigned errors = check(test);
    total_errors += errors;
    nr_puts("[tilelang-bosc] case=0x");
    nr_hex32((uint32_t)test);
    nr_puts(" errors=0x");
    nr_hex32(errors);
    nr_puts("\r\n");
  }
  nr_puts(total_errors ? "TileLang BOSC GEMM: FAIL\r\n"
                       : "TileLang BOSC GEMM: PASS\r\n");
  return total_errors ? 1 : 0;
}
