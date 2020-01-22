#ifndef READ_SYS_H
#define READ_SYS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t* read_sys(uint32_t* N, uint32_t* M);

uint32_t* pack_sys_data(const uint32_t* data, uint32_t N, uint32_t M);

#ifdef __cplusplus
}
#endif

#endif /* READ_SYS_H */
