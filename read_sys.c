// MIT License
//
// Copyright (c) 2018 Kai-Chun Ning <kaichun.ning@gmail.com>,
//                    Ruben Niederhagen, Richard Petri
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#define _POSIX_C_SOURCE 200809L

#include "read_sys.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* for parsing challenge file */
const char* CHA_GF_LINE = "Galois Field";
const char* CHA_VAR_LINE = "Number of variables";
const char* CHA_EQ_LINE = "Number of polynomials";
const char* CHA_SEED_LINE = "Seed";
// const char* CHA_ORD_LINE = "Order";
const char* CHA_EQ_START = "*********";
const size_t MAX_PRE_LEN = 128;

/* testing if pre is a prefix of the string */
static inline bool check_prefix(const char* pre, const char* str)
{
  return !strncmp(pre, str, strnlen(pre, MAX_PRE_LEN));
}

/* parse the header of challenge file, return true is still in header.
 * return false otherwise.
 */
static bool parse_cha_header(const char* str, uint32_t* N, uint32_t* M)
{
  bool verbose = false;
  if (check_prefix(CHA_EQ_START, str)) {
    if (verbose) {
      printf("\t\treading equations...\n");
    }
    return false;
  }

  uint64_t var_num, eq_num, seed;

  if (check_prefix(CHA_VAR_LINE, str)) {
    if (1 != sscanf(str, "%*s %*s %*s %*s : %" PRIu64, &var_num)) {
      fprintf(stderr, "[!] cannot parse number of unknowns: %s\n", str);
      exit(-1);
    }

    *N = var_num;

    //    if (var_num != N)
    //    {
    //      fprintf(stderr, "Number of variables in input file does not fit
    //      compile options!\n"); fprintf(stderr, "%" PRIu64 " != %i\n",
    //      var_num, N); exit(-1);
    //    }

    if (verbose) {
      printf("\t\tnumber of variables: %" PRIu64 "\n", var_num);
    }

  } else if (check_prefix(CHA_EQ_LINE, str)) {
    if (1 != sscanf(str, "%*s %*s %*s %*s : %" PRIu64, &eq_num)) {
      fprintf(stderr, "[!] cannot parse number of equations: %s\n", str);
      exit(-1);
    }

    *M = eq_num;

    //    if (eq_num != M)
    //    {
    //      fprintf(stderr, "Number of equations in input file does not fit
    //      compile options!\n"); fprintf(stderr, "%" PRIu64 " != %i\n", eq_num,
    //      M); exit(-1);
    //    }

    if (verbose) {
      printf("\t\tnumber of equations: %" PRIu64 "\n", eq_num);
    }

  } else if (check_prefix(CHA_SEED_LINE, str)) {
    if (1 != sscanf(str, "%*s : %" PRIu64, &seed)) {
      fprintf(stderr, "[!] unable to seed: %s\n", str);
      exit(-1);
    }

    if (verbose) {
      printf("\t\tseed: %" PRIu64 "\n", seed);
    }

  } else if (check_prefix(CHA_GF_LINE, str)) {
    int prime = 0;
    if ((1 != sscanf(str, "%*s %*s : GF(%d)", &prime)) || prime != 2) {
      fprintf(stderr, "[!] unable to process GF(%d)\n", prime);
      exit(-1);
    }

    if (verbose) {
      printf("\t\tfield: GF(%d)\n", prime);
    }
  }

  return true;
}

/* parse the system of challenge file. Note this will destroy the string */
static void
parse_cha_eqs(char* str, const uint64_t eq_idx, uint32_t* orig_sys, uint32_t N)
{
  char* ptr = NULL;

  uint64_t i = 0;
  ptr = strtok(str, " ;");
  while (NULL != ptr) {
    orig_sys[(N * (N - 1) / 2 + N + N + 1) * eq_idx + i] = atoi(ptr);
    i += 1;
    ptr = strtok(NULL, " ;\n");
  }
}

uint32_t* read_sys(uint32_t* N, uint32_t* M)
{
  FILE* fp = stdin;
  // FILE* fp = fopen( "data.in" , "r");

  // NOTE: expand the buffer if needed
  const size_t buf_size = 0x1 << 20; // 1MB per line
  char* buf = (char*)malloc(buf_size);
  uint64_t eq_idx = 0;

  while (NULL != fgets(buf, buf_size, fp)) {
    if (!parse_cha_header(buf, N, M))
      break;
  }

  if (feof(fp)) {
    free(buf);

    return NULL;
  }

  uint32_t* data = (uint32_t*)malloc(((*N) * ((*N) - 1) / 2 + 2 * (*N) + 1) *
                                     (*M) * sizeof(uint32_t));

  for (int i = 0; i < *M; i++) {
    if (NULL != fgets(buf, buf_size, fp)) {
      parse_cha_eqs(buf, eq_idx++, data, *N);
    } else {
      free(buf);
      free(data);

      fprintf(stderr, "Error while reading input data!\n");
      exit(-1);
    }
  }

  if (feof(fp))
    fprintf(stderr, "end of file\n");

  // fclose(fp);
  free(buf);

  return data;
}

uint32_t* pack_sys_data(const uint32_t* data, uint32_t N, uint32_t M)
{
  //  reduce input system - remove squares

  uint32_t num_blocks = ((M >> 5) + ((M & 31) == 0 ? 0 : 1));

  uint32_t* sys = (uint32_t*)malloc(sizeof(uint32_t) *
                                    (N * (N - 1) / 2 + N + 1) * num_blocks);
  uint32_t* sq0 = (uint32_t*)malloc(sizeof(uint32_t) * N * num_blocks);

  int sq_id = 0;

  int is = 0;
  int id = 0;

  for (int v0 = 0; v0 < N; v0++) {
    for (int v1 = 0; v1 <= v0; v1++) {
      for (uint32_t b = 0; b < M; b += 32) {
        uint32_t val = 0;

        for (int j = (((M - b) >= 32) ? b + 31 : (M - 1)); j >= (int)b; j--)
          val = (val << 1) | data[(N * (N - 1) / 2 + N + N + 1) * j + is];

        if (v0 == v1)
          sq0[sq_id + N * (b >> 5)] = val;
        else
          sys[(N * (N - 1) / 2 + N + 1) * (b >> 5) + id] = val;
      }

      is += 1;

      if (v0 == v1)
        sq_id += 1;
      else
        id += 1;
    }
  }

  for (int v0 = 0; v0 < N; v0++) {
    for (uint32_t b = 0; b < M; b += 32) {
      uint32_t val = 0;

      for (int j = (((M - b) >= 32) ? b + 31 : (M - 1)); j >= (int)b; j--)
        val = (val << 1) | data[(N * (N - 1) / 2 + N + N + 1) * j + is];

      sys[(N * (N - 1) / 2 + N + 1) * (b >> 5) + id] =
          val ^ sq0[v0 + N * (b >> 5)];
    }

    is += 1;
    id += 1;
  }

  {
    for (uint32_t b = 0; b < M; b += 32) {
      uint32_t val = 0;

      for (int j = (((M - b) >= 32) ? b + 31 : (M - 1)); j >= (int)b; j--)
        val = (val << 1) | data[(N * (N - 1) / 2 + N + N + 1) * j + is];

      sys[(N * (N - 1) / 2 + N + 1) * (b >> 5) + id] = val;
    }
  }

  free(sq0);

  return sys;
}
