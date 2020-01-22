#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "guess.h"
#include "read_sys.h"

int main(int argc, char** argv)
{
  if (argc > 1) {
    setDevice(atoi(argv[1]));
  }
  while (1) {
    uint32_t N;
    uint32_t M;

    uint32_t* data = read_sys(&N, &M);

    if (data == NULL) {
      break;
    }

    uint64_t sol = searchSolution(data, N, M);

    if (sol < UINT64_MAX)
      printf("%0lX\n", sol);

    // printf("done\n");
    fflush(stdout);

    free(data);
  }

  return 0;
}
