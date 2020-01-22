#ifndef CHECK_THREAD_H
#define CHECK_THREAD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t* check_thread(const uint32_t* deg2, const uint32_t* deg1,
                       uint32_t thread, uint32_t N, uint32_t K);

#ifdef __cplusplus
}
#endif

#endif /* CHECK_THREAD_H */
