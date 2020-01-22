#include "partial_eval.h"

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

static void deg0_coefs(const uint32_t* deg2, const uint32_t* deg1,
                       uint32_t* result, uint32_t N, uint32_t K)
{
  uint32_t rounds;
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t z = 0;
  uint32_t tmp = 0;

  uint32_t diff[N - K];

  diff[0] = deg1[0];

  for (int i = 1; i < (N - K); i++) {
    diff[i] = deg1[i] ^ deg2[COEF(i - 1, i)];
  }

  uint32_t res = deg1[N - K];

  result[0] = res;

  for (rounds = 1; rounds < (1 << (N - K)); rounds += 1) {
    tmp = (rounds & (rounds - 1));
    y = rounds ^ tmp;
    x ^= y;
    z = tmp ^ (tmp & (tmp - 1));

    uint32_t y_pos = cnt0(y);
    uint32_t z_pos = cnt0(z);

    if (z_pos > y_pos)
      diff[y_pos] ^= deg2[COEF(y_pos, z_pos)];

    res ^= diff[y_pos];
    result[x] = res;
    tmp = (y_pos * (y_pos - 1)) >> 1;
  }
}

static void
deg1_coefs(const uint32_t* deg1, uint32_t* result, uint32_t N, uint32_t K)
{
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t res = deg1[N - K];

  result[0] = res;

  for (uint32_t rounds = 1; rounds < (1 << (N - K)); rounds += 1) {
    y = rounds ^ (rounds & (rounds - 1));
    x ^= y;

    res ^= deg1[cnt0(y)];
    result[x] = res;
  }
}

void partial_eval(const uint32_t* sys, uint32_t* deg1, uint32_t N, uint32_t K)
{
  uint32_t deg1_sys[(N - K) + 1];
  uint32_t deg2_sys[COEF(N - K, N - K) + 1];
  uint64_t pos = 0;

  // deg2 part
  for (uint32_t i = 0; i < K; i++) {
    for (uint32_t j = 0; j <= (N - K); j++) {
      deg1_sys[j] = sys[COEF(0, j + K) + i];
    }

    deg1_coefs(deg1_sys, &deg1[pos], N, K);
    pos += (1 << (N - K));
  }

  // deg1 part
  for (uint32_t j = 1; j <= (N - K); j++) {
    for (uint32_t i = 0; i <= j; i++) {
      deg2_sys[COEF(i, j)] = sys[COEF(i + K, j + K)];
    }
  }

  deg0_coefs(deg2_sys, deg2_sys + COEF(0, N - K), &deg1[pos], N, K);
}
