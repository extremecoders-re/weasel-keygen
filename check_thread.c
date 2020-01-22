#include "check_thread.h"

#include <stdlib.h>

#define NUM_THREADS (1 << (N - K))

#define COEF(I, J) ((((J) * ((J)-1)) >> 1) + (I))

#ifndef HAVE_CNT
static const int MultiplyDeBruijnBitPosition[32] = {
    0,  1,  28, 2,  29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4,  8,
    31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6,  11, 5,  10, 9};

static uint32_t cnt0(uint32_t v)
{
  return MultiplyDeBruijnBitPosition[((uint32_t)((v & -v) * 0x077CB531U)) >>
                                     27];
}
#define HAVE_CNT
#endif

uint32_t* check_thread(const uint32_t* deg2, const uint32_t* deg1,
                       uint32_t thread, uint32_t N, uint32_t K)
{
  uint32_t rounds;
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t z = 0;
  uint32_t tmp = 0;
  uint32_t count = 0;

  uint32_t diff[K];

  uint32_t* result = (uint32_t*)malloc((1 << K) * sizeof(uint32_t));

  diff[0] = deg1[0 * NUM_THREADS + thread];

  for (int i = 1; i < K; i++) {
    diff[i] = deg1[i * NUM_THREADS + thread] ^ deg2[COEF(i - 1, i)];
  }

  uint32_t res = deg1[K * NUM_THREADS + thread];

  for (rounds = 1; rounds < (1 << K); rounds += 1) {
    tmp = (rounds & (rounds - 1));
    y = rounds ^ tmp;
    x ^= y;
    z = tmp ^ (tmp & (tmp - 1));

    uint32_t y_pos = cnt0(y);
    uint32_t z_pos = cnt0(z);

    if (z_pos > y_pos)
      diff[y_pos] ^= deg2[COEF(y_pos, z_pos)];

    res ^= diff[y_pos];
    if (res == 0)
      result[count++] = x;
  }

  result[count] = 0;

  return result;
}
