#ifndef PARTIAL_EVAL_H
#define PARTIAL_EVAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void partial_eval(const uint32_t* sys, uint32_t* deg1, uint32_t N,
                  uint32_t locK);

#ifdef __cplusplus
}
#endif

#endif /* PARTIAL_EVAL_H */
