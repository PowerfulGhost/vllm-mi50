#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

__global__ void quantize_q8_0_mi50(half *__restrict__ x,
                                   block_q8_0 *__restrict__ out, int nrows_x,
                                   int ncols_x, int block_q8_0_per_row)
{
    /*
    quantize x to q8_0 on gcn arch

    x: input tensor (fp16) need to be quantized
    out: quantized block
    nrows_x: # of rows of x
    ncols_x: # of cols of x
    block_q8_0_per_row: # of block_q8_0 per row

    expecting to be launched with:
    - gridDim = (nrows_x, (ncols_x + warpSize-1) / warpSize)
    - blockDim = (warpSize, 1)

    assuming ncols_x % 32 == 0

    each thread handle to a element in the input

    each block_q8_0 contains 32 elements, so 1 warp on gcn (64 threads) can handle 2 block_q8_0

    in each warp:
    1. load 64 elements (2 block_q8_0s) from x
    2. calculate delta(d) of first 32-elements and second 32-elements
    which corresponding to 1st block_q8_0 and 2nd block_q8_0
    d = max(x_i) / 127
    3. calculate 64 quantized elements(qs) with d
    qs = x_i / d = x_i * 127 / max
    4. write qs and d to output
    */

    int global_thread_idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int global_thread_idx_y = blockDim.y * blockIdx.y + threadIdx.y;

    // 1. load element from x
    float xi = 0.0f;
    if (global_thread_idx_x < ncols_x) // not pad elements
    {
        xi = __half2float(x[ncols_x * global_thread_idx_y + global_thread_idx_x]);
    }

    // 2. calculate d and sum
    float max = fabsf(xi);
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        // broadcast max and sum to all threads handling the same block_q8_0
        max = fmaxf(max, __shfl_xor_sync((uint64_t)-1, max, offset, 32));
    }
    float delta = max / 127.0f;

    // 3. calculate qs with d
    int8_t qs = 0;
    if (delta != 0.0f)
    {
        float q = __half2float(xi) / __half2float(delta);
        qs = (int8_t)roundf(q);
    }

    // 4. write qs and d to output
    block_q8_0 *b = out +
                    blockIdx.y * block_q8_0_per_row +
                    blockIdx.x * 2 +  // each block handle 2 block_q8_0
                    threadIdx.x / 32; // 0 or 1, 1st or 2nd block_q8_0 of this block
    int qs_idx = threadIdx.x % 32;
    if (global_thread_idx_x < ncols_x)
    {
        b->qs[qs_idx] = qs;
        if (qs_idx == 0)
        {
            b->d = __float2half(delta);
        }
    }
}