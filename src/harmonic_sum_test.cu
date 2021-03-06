// Ewan Barr 2015 (@GTC15)
// Refined harmonic summing kernels
// Compile with:
// nvcc -O3 -Xptxas -dlcm=ca -m64 -arch=sm_35 -keep -o <> <>.cu

#include "harmonic_sum.cuh"
#include "cuda_errors.cuh"
#define MAX_HARMS 5
#define MAX_SIZE 4194304
#define ITERATIONS 100

int main(void)
{

  harmsum_tex.filterMode = cudaFilterModePoint;

  float elapsedTime,bestTime;
  int bestThreads;
  cudaEvent_t start,stop;
  float* hsum_input;
  float** hsum_output;
  float** hsum_output_h;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //Allocate all memory for tests
  abort_on_cuda_fail( cudaMalloc((void**)&hsum_input, sizeof(float) * MAX_SIZE));
  abort_on_cuda_fail( cudaMallocHost((void**)&hsum_output_h, sizeof(float*)*MAX_HARMS));
  abort_on_cuda_fail( cudaMalloc((void**)&hsum_output, sizeof(float*)*MAX_HARMS));
  for (int ii=0;ii<MAX_HARMS;ii++)
    abort_on_cuda_fail( cudaMalloc((void**)&(hsum_output_h[ii]), sizeof(float) * MAX_SIZE));
  abort_on_cuda_fail( cudaMemcpy(hsum_output,hsum_output_h,sizeof(float*)*MAX_HARMS,cudaMemcpyHostToDevice) );

  //Speed tests
  printf("Benchmarking harmonic sum algorithm with %d point array\n",MAX_SIZE);
  for (int harm=1;harm<6;harm++)
    {
      bestTime = 99999;
      bestThreads = 0;
      for (int nthreads=64; nthreads<1025; nthreads+=64)
        {
          cudaEventRecord(start, 0);
          for (int ii=0;ii<ITERATIONS;ii++)
            incoherent_harmonic_sum_helper(hsum_input,hsum_output,MAX_SIZE,harm,nthreads);
          abort_on_cuda_fail( cudaStreamSynchronize(0) );
          cudaEventRecord(stop, 0);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&elapsedTime, start, stop);
          if (elapsedTime < bestTime){
	    bestTime = elapsedTime;
	    bestThreads = nthreads;
	  }
          printf("Nthreads: %d   Nharms: %d  Benchmark: %f ms\n", nthreads,1<<harm,elapsedTime/ITERATIONS);
	}
      printf("\nBest thread count for %d harmonics: %d  (time: %f ms) \n\n",1<<harm,bestThreads,bestTime/ITERATIONS);
    }

  //clean up
  abort_on_cuda_fail( cudaMemcpy(hsum_output_h,hsum_output,sizeof(float*)*MAX_HARMS,cudaMemcpyDeviceToHost) );
  for (int ii=0;ii<MAX_HARMS;ii++)
    cudaFree(hsum_output_h[ii]);
  cudaFree(hsum_output);
  cudaFree(hsum_input);
  cudaFreeHost(hsum_output_h);      
}
