SHELL=/bin/bash

MAX_K = 20  # maximum search range per GPU thread

UNROLL = 8 # unroll GPU kernel loop

NVCC = nvcc
CC = gcc
CXX = g++

COMMON_FLAGS = -Wall
CFLAGS = -O3 -fPIC -std=c11 $(COMMON_FLAGS)
CXXFLAGS = -O3 -fPIC -std=c++11 $(COMMON_FLAGS)
NVCFLAGS = -O3 -DMAX_K=$(MAX_K) --std=c++11 $(foreach option, $(COMPILER_OPTIONS), --compiler-options $(option))
CUDA_ARCH = -arch=sm_30\
	-gencode=arch=compute_30,code=sm_30 \
	-gencode=arch=compute_50,code=sm_50 \
	-gencode=arch=compute_52,code=sm_52 \
	-gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 \
	-gencode=arch=compute_62,code=sm_62 \
	-gencode=arch=compute_70,code=sm_70 \
	-gencode=arch=compute_70,code=compute_70
NVCFLAGS += $(CUDA_ARCH)
CUDA_LDFLAGS = -L/usr/local/cuda/lib64 -lcudart_static
LDFLAGS = -lpthread -ldl -lrt

CUDA_SRC = guess.cu
C_SRC = check_sol.c partial_eval.c check_thread.c read_sys.c
OBJ = $(CUDA_SRC:.cu=.co) $(C_SRC:.c=.o) $(CXX_SRC:.cpp=.o)

BIN = guess

%.d: %.c
	$(CC) $(CFLAGS) -MM -o $@ $<

%.d: %.cpp
	$(CXX) $(CXXFLAGS) -MM -o $@ $<

%.d: %.cu
	$(NVCC) -M -o $@ $<

default: $(BIN)

kernel.inc: gen_kernel.py
	./gen_kernel.py --unroll=$(UNROLL) > kernel.inc

guess.cu: kernel.inc

-include $(C_SRC:.c=.d)
-include $(CXX_SRC:.cpp=.d)
-include $(CUDA_SRC:.cu=.d)

%.co: %.cu
	$(NVCC) -c $(NVCFLAGS) $< -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

guess: main.o $(OBJ)
	$(CXX) $^ $(CUDA_LDFLAGS) $(LDFLAGS) -o guess


test-large: guess gen_sys.py fix.py
	./gen_sys.py -m 48 -n 42 > /tmp/full; for fix in 00 01 10 11; do cat /tmp/full | ./fix.py -f $$fix; done | ./guess 0

test-small: guess gen_sys.py fix.py
	./gen_sys.py -m 56 -n 40 | ./guess 0


clean:
	rm -f *.o *.d *.co $(BIN) kernel.inc
