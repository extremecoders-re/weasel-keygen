#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <cuda.h>

#ifndef CUDA_ASSERT
#include <stdio.h>
#define CUDA_ASSERT(ans)                                                       \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void
gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
#endif

template <class Type> class cudaData
{
public:
  cudaData(size_t len_host, size_t len_dev = 0);
  ~cudaData();

  size_t size_host();
  size_t size_dev();
  void clear();
  void write(size_t off_src = 0, size_t size = 0, size_t off_des = 0);
  void read(size_t off_src = 0, size_t size = 0, size_t off_des = 0);

  Type* host;
  Type* dev;

private:
  size_t sz_host;
  size_t sz_dev;
};

template <class Type> cudaData<Type>::cudaData(size_t len_host, size_t len_dev)
{
  if (len_dev == 0)
    len_dev = len_host;

  sz_host = len_host * sizeof(Type);
  sz_dev = len_dev * sizeof(Type);

  host = (Type*)malloc(sz_host);

  CUDA_ASSERT(cudaMalloc((void**)&dev, sz_dev));
}

template <class Type> cudaData<Type>::~cudaData()
{
  free(host);
  cudaFree(dev);
}

template <class Type> size_t cudaData<Type>::size_host() { return sz_host; }

template <class Type> size_t cudaData<Type>::size_dev() { return sz_dev; }

template <class Type> void cudaData<Type>::clear() { memset(host, 0, sz_host); }

template <class Type>
void cudaData<Type>::write(size_t off_src, size_t size, size_t off_des)
{
  if (size == 0)
    size = (sz_host <= sz_dev) ? sz_host : sz_dev;

  CUDA_ASSERT(
      cudaMemcpy(&dev[off_src], &host[off_des], size, cudaMemcpyHostToDevice));
}

template <class Type>
void cudaData<Type>::read(size_t off_src, size_t size, size_t off_des)
{
  if (size == 0)
    size = (sz_host <= sz_dev) ? sz_host : sz_dev;

  CUDA_ASSERT(
      cudaMemcpy(&host[off_src], &dev[off_des], size, cudaMemcpyDeviceToHost));
}

#endif
