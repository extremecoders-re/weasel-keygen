#ifndef GUESS_H
#define GUESS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

double get_ms_time(void);

void setDevice(int device);

uint64_t
searchSolution(uint32_t* coefficients, unsigned int number_of_variables,
               unsigned int number_of_equations);

#ifdef __cplusplus
}
#endif

#endif /* GUESS_H */
