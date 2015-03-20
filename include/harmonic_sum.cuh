/* 
 * Ewan Barr 2015 (GTC15)
 */
#include "cufft.h"
#define ONE_OVER_SQRT2 0.70710678118654746f
#define ONE_OVER_SQRT4 0.5f
#define ONE_OVER_SQRT8 0.35355339059327373f
#define ONE_OVER_SQRT16 0.25f
#define ONE_OVER_SQRT32 0.17677669529663687f

//Define a global texture for harmonic sum texture loads
texture<cufftReal, 1, cudaReadModeElementType> harmsum_tex;

/* 
 * Custom texture fetch that exploits the mantissa 
 * in the floating point to do float to int switching.
 */
__forceinline__ __device__ float custom_fetch(
   const unsigned int idx,
   float fraction)
{
  return tex1Dfetch(harmsum_tex,__float_as_int(__int_as_float(idx)*fraction));
}

/*
 * Incoherent harmonic sum kernel. Speed improved by
 * templating maximum harmonic sum. Plan was to do 
 * nested loop unrolls, but nvcc refuses to do it
 * properly, so resorting to hand rolling the outer 
 * loop.
 */
template <unsigned int max_harm>
__global__ void incoherent_harmonic_sum(
    float** d_odata,
    unsigned int size)
{
  const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (idx>=size) return;
  
  int ii;
  float val = tex1Dfetch(harmsum_tex,idx);

  if (max_harm > 0){
    val += custom_fetch(idx,0.5f);
    d_odata[0][idx] = val*ONE_OVER_SQRT2;
  }

  if (max_harm > 1){
#pragma unroll
    for (ii=1;ii<4;ii+=2)
      val += custom_fetch(idx,ii/4.0f);
    d_odata[1][idx] = val*ONE_OVER_SQRT4;
  }

  if (max_harm > 2){
#pragma unroll
    for (ii=1;ii<8;ii+=2)
      val += custom_fetch(idx,ii/8.0f);
    d_odata[2][idx] = val*ONE_OVER_SQRT8;
  }

  if (max_harm > 3){
#pragma unroll
    for (ii=1;ii<16;ii+=2)
      val += custom_fetch(idx,ii/16.0f);
    d_odata[3][idx] = val*ONE_OVER_SQRT16;
  }
  
  if (max_harm > 4){
#pragma unroll
    for (ii=1;ii<32;ii+=2)
      val += custom_fetch(idx,ii/32.0f);
    d_odata[3][idx] = val*ONE_OVER_SQRT32;
  }
}

// Helper function for launching from host
void incoherent_harmonic_sum_helper(float* input, float** output, unsigned int size, unsigned int nharms, unsigned int nthreads)
{
  cudaBindTexture(0, harmsum_tex, input, size*sizeof(cufftReal));
  
  unsigned int nblocks = size/nthreads + 1;

  switch ( nharms ) 
    {
    case 1:
      incoherent_harmonic_sum<1><<<nblocks,nthreads>>>(output,size);
      break;
    case 2:
      incoherent_harmonic_sum<2><<<nblocks,nthreads>>>(output,size);
      break;
    case 3:
      incoherent_harmonic_sum<3><<<nblocks,nthreads>>>(output,size);
      break;
    case 4:
      incoherent_harmonic_sum<4><<<nblocks,nthreads>>>(output,size);
      break;
    case 5:
      incoherent_harmonic_sum<5><<<nblocks,nthreads>>>(output,size);
      break;
    default:
      return;
    }
  
  cudaUnbindTexture(harmsum_tex);
  
}
