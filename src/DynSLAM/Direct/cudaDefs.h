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

// TODO(andrei): If integrating direct alignment in codebase permanently, resolve duplication
// between this and ITM.
#ifndef _CPU_AND_GPU_CODE_
  #define _CPU_AND_GPU_CODE_ __device__ __host__ // for CUDA device code
#endif

#else   // ndef COMPILE_WITH_CUDA

#ifndef _CPU_AND_GPU_CODE_
  #define _CPU_AND_GPU_CODE_
#endif

#endif  // ndef COMPILE_WITH_CUDA
