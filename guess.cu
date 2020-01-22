#include "check_sol.h"
#include "check_thread.h"
#include "guess.h"
#include "partial_eval.h"
#include "read_sys.h"

#include "cuda_util.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define NUM_THREADS (1 << (N - K))
#define BLOCK_DIM (NUM_THREADS > 128 ? 128 : NUM_THREADS)
#define GRID_DIM (NUM_THREADS / BLOCK_DIM)

#define PRINT_SOL(X) printf("%lX\n", X)
// #define PRINT_SOL(X)

#define LOG(level, f_, ...) fprintf(stderr, (f_), ##__VA_ARGS__)
// #define LOG(level, f_, ...)

extern "C" double get_ms_time(void)
{
  struct timeval timev;

  gettimeofday(&timev, NULL);
  return (double)timev.tv_sec * 1000 + (double)timev.tv_usec / 1000;
}

__device__ __constant__ uint32_t deg2_block[MAX_K * (MAX_K - 1) / 2];

__global__ void guess(uint32_t* deg1, uint32_t* result, uint32_t num_threads);

#include "kernel.inc"

static int cuda_device = 0;
static bool init = false;

extern "C" void setDevice(int device)
{
  cuda_device = device;
  init = false;
}

extern "C" uint64_t
searchSolution(uint32_t* coefficients, unsigned int number_of_variables,
               unsigned int number_of_equations)
{

  if (!init) {
    double initTime = 0;
    initTime -= get_ms_time();

    // set to designated device
    // int test;
    CUDA_ASSERT(cudaSetDevice(cuda_device));
    // cudaGetDevice(&test);
    // assert(atoi(argv[1]) == test);

    initTime += get_ms_time();
    LOG(INFO, "init time = %f\n", initTime);

    init = true;
  }

  double preTime = 0, memTime = 0, recvTime = 0, checkTime = 0, ctTime = 0;
  float kernelTime = 0;
  uint32_t solCount = 0, ctCount = 0;

  uint64_t res = UINT64_MAX;

  // create events here
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  uint32_t N = number_of_variables;
  uint32_t M = number_of_equations;

  uint32_t K = 32;

  if (K > MAX_K)
    K = MAX_K;

  if (N <= K)
    K = N - 1;

  uint32_t* sys = pack_sys_data(coefficients, N, M);

  preTime -= get_ms_time(); // partial evaluation

  cudaData<uint32_t> deg1((K + 1) * NUM_THREADS);

  partial_eval(sys, deg1.host, N, K);

  preTime += get_ms_time();

  memTime -= get_ms_time(); // initializing GPU memory space

  // initialize constant memory space for the quadratic part
  CUDA_ASSERT(
      cudaMemcpyToSymbol(deg2_block, sys, sizeof(uint32_t) * K * (K - 1) / 2));

  // initialize global memory space for the linear parts
  deg1.write();

  // initialize global memory space for the results of each threads
  cudaData<uint32_t> result(NUM_THREADS);

  memTime += get_ms_time();

  // launch kernel function and measure the elapsed time
  cudaEventRecord(start, 0);

  guess<<<GRID_DIM, BLOCK_DIM>>>(deg1.dev, result.dev, NUM_THREADS, K);

  CUDA_ASSERT(cudaEventRecord(stop, 0));
  CUDA_ASSERT(cudaEventSynchronize(stop));

  CUDA_ASSERT(cudaEventElapsedTime(&kernelTime, start, stop));

  recvTime -= get_ms_time(); // copy the results of each thread to host

  result.read();

  recvTime += get_ms_time();

  checkTime -= get_ms_time(); // check if the results are available

  int32_t ans;

  for (uint64_t i = 0; i < NUM_THREADS; i++) {
    ans = result.host[i];

    if (ans) {
      solCount++;

      if (ans & 0x80000000) // more than one solution
      {
        uint32_t* sols;

        ctCount++;
        ctTime -= get_ms_time();
        sols = check_thread(sys, deg1.host, i, N, K);
        ctTime += get_ms_time();

        uint32_t j;

        for (j = 0; sols[j]; j++) {
          if (check_sol(sys, (i << K) | sols[j], N, M) == 1) {
            LOG(INFO, "thread %lX ---------> solution %X\n", i, sols[j]);
            PRINT_SOL((i << K) | sols[j]);

            res = (i << K) | sols[j];

            goto end;
          }
        }

        LOG(INFO, "thread %lX ---------> several solutions: %u\n", i, j);

        free(sols);
      } else // only one solution
      {
        if (check_sol(sys, (i << K) | ans, N, M) == 0) {
          LOG(INFO, "thread %lX ---------> one solution %X\n", i, ans);
          PRINT_SOL((i << K) | ans);

          res = (i << K) | ans;

          goto end;
        }
      }
    }

    if (deg1.host[K * NUM_THREADS + i] ==
        0) // special case: check for (prtial) zero solution
    {
      if (check_sol(sys, (i << K) | 0, N, M) == 0) {
        LOG(INFO, "thread %lX ---------> one solution 0\n", i);
        PRINT_SOL(i << K);

        res = (i << K);

        goto end;
      }
    }
  }

end:

  checkTime += get_ms_time();

  // print the time for each step
  LOG(INFO, "partial ");
  LOG(INFO, "mem ");
  LOG(INFO, "kernel ");
  LOG(INFO, "recv ");
  LOG(INFO, "check #sol ");
  LOG(INFO, "(mult sol: t #ct)\n");
  LOG(INFO, "%.3f ", preTime);
  LOG(INFO, "%.3f ", memTime);
  LOG(INFO, "%.3f ", kernelTime);
  LOG(INFO, "%.3f ", recvTime);
  LOG(INFO, "%.3f ", checkTime);
  LOG(INFO, "%u ", solCount);
  LOG(INFO, "(%.3f  %u)\n", ctTime, ctCount);

  // release memory spaces
  free(sys);

  return res;
}
