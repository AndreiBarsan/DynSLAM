#ifndef COMPILE_WITHOUT_CUDA
  
  #include <cuda_runtime.h>
  
  #define CUDA_SAFE_CALL(call){                                                         \
      cudaError_t err = call;                                                         \
      if(err!=cudaSuccess)                                                              \
      {                                                                               \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err)); \
	exit(0);                                                                      \
      }                                                                               \
  }
  
  #define _CPU_AND_GPU_CODE_ __device__ __host__ // for CUDA device code

#else
  
  #define _CPU_AND_GPU_CODE_

#endif