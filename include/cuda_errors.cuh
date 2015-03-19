#ifndef CUDA_ERRORS_CUH_
#define CUDA_ERRORS_CUH_

#include "stdio.h"
#include "cuda.h"
#include "cufft.h"

#define abort_on_cuda_fail(code) { catch_cuda_error((code), __FILE__, __LINE__); }
#define abort_on_cufft_fail(code) { catch_cufft_error((code), __FILE__, __LINE__); }

static const char * cufftGetErrorString(cufftResult error)
{
  switch (error)
    {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
    }

  return "<unknown>";
}

inline void catch_cuda_error(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"catch_cuda_error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

inline void catch_cufft_error(cufftResult code, const char *file, int line, bool abort=true)
{
  if (code != CUFFT_SUCCESS)
    {
      fprintf(stderr,"catch_cufft_error: %s %s %d\n", cufftGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

#endif
