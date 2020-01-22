#include "check_sol.h"

uint32_t check_sol(const uint32_t* sys, uint64_t sol, uint32_t N, uint32_t M)
{
  uint32_t i, j, pos = 0;
  uint32_t x[N], check = 0;

  for (uint32_t b = 0; b < M; b += 32) {
    uint32_t mask = (M - b) >= 32 ? 0xffffffff : ((1 << (M - b)) - 1);

    for (i = 0; i < N; i++)
      x[i] = ((sol >> i) & 1) ? mask : 0;

    // computing quadratic part
    for (j = 1; j < N; j++)
      for (i = 0; i < j; i++)
        check ^= sys[pos++] & x[i] & x[j];

    // computing linear part
    for (i = 0; i < N; i++)
      check ^= sys[pos++] & x[i];

    // constant part
    check ^= sys[pos++];
  }

  return check;
}
