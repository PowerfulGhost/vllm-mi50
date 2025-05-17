#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define WARP_SIZE_GCN 64
#define SHARED_MEMORY_ROW_SIZE_BYTE 128

// matmul tile size def
#define TILE_SIZE_COL 4 // unit: (unquantized) element
#define TILE_SIZE_ROW 4 //  unit: (unquantized) element

// threads per block
#define THREADS_PER_BLOCK WARP_SIZE_GCN *TILE_SIZE_COL *TILE_SIZE_ROW

// how many quantize blocks (block_q8_0) in each segment
// for each tile, input data is loaded and calculated segment by segment since shared memory is limited.
// NOTE: actually we need to load double number of qblocks, half of them from qweight and half of them from x.
// NOTE: each warp handle one segment at same time, since each thread handle one int8x4 dot product, each warp can handle 64/8 qblocks
#define QBLOCKS_PER_SEGMENT (WARP_SIZE_GCN / 8)

__device__ __forceinline__ void allocate_shared_memory(
    int32_t *&smem_qweight_segment_qs,
    half *&smem_qweight_segment_d,
    int32_t *&smem_x_segment_qs,
    half *&smem_x_segment_d)
{
  __shared__ int32_t qweight_qs[TILE_SIZE_COL * QBLOCKS_PER_SEGMENT * QI8_0];
  __shared__ int32_t x_qs[TILE_SIZE_ROW * QBLOCKS_PER_SEGMENT * QI8_0];

  __shared__ half qweight_d[TILE_SIZE_COL * QBLOCKS_PER_SEGMENT];
  __shared__ half x_d[TILE_SIZE_ROW * QBLOCKS_PER_SEGMENT];

  smem_qweight_segment_qs = qweight_qs;
  smem_x_segment_qs = x_qs;

  smem_qweight_segment_d = qweight_d;
  smem_x_segment_d = x_d;
}

__device__ __forceinline__ void load_int8x4(block_q8_0 *qblock_start, int32_t *smem_qs_start,
                                            const int &global_out_col_idx, const int &global_out_row_idx, const int &qblocks_per_row,
                                            const int &tile_col_or_row_idx,
                                            const int &segment_idx, const int &qblock_idx_in_segment, const int &int8x4_idx_in_qblock)
{
  int qblock_idx = global_out_row_idx * qblocks_per_row + global_out_col_idx;
  block_q8_0 *qblock = qblock_start + qblock_idx;

  int smem_qs_idx = (tile_col_or_row_idx * QBLOCKS_PER_SEGMENT + qblock_idx_in_segment) * 8 + int8x4_idx_in_qblock;
  int32_t *smem_qs = smem_qs_start + smem_qs_idx;

  *smem_qs = *reinterpret_cast<int32_t *>(qblock->qs + int8x4_idx_in_qblock);
}

__device__ __forceinline__ void load_d(block_q8_0 *qblock_start, half *smem_d_start,
                                       const int &global_out_col_idx, const int &global_out_row_idx, const int &qblocks_per_row,
                                       const int &tile_col_or_row_idx,
                                       const int &segment_idx, const int &qblock_idx_in_segment)
{
  int qblock_idx = global_out_row_idx * qblocks_per_row + global_out_col_idx;
  block_q8_0 *qblock = qblock_start + qblock_idx;

  int smem_d_idx = tile_col_or_row_idx * QBLOCKS_PER_SEGMENT + qblock_idx_in_segment;
  half *smem_d = smem_d_start + smem_d_idx;

  *smem_d = qblock->d;
}

__global__ void mul_mat_q8_0_mi50(
    block_q8_0 *__restrict__ qweight,
    block_q8_0 *__restrict__ qx,
    half *__restrict__ out,
    int ncols,
    int nrows_weight,
    int nrows_x,
    int num_qblocks_per_row,
    int num_segments_per_row,
    int num_invald_qblocks_per_row)
{
  /*
  Arguments:
    qweight: quantized weight, q8_0.
    qx: quantiezd input, q8_0.
    out: output, half.
    ncols: num of columns of weight(unquantized) and x(unquantized), assuming
      they are equal.
    nrows_weight: num of rows of weight(unquantized).
    nrows_x: num of rows of x(unquantized).
    num_qblocks_per_row: num of block_q8_0 per row, equal to ceil(ncols/QK8_0).
    num_segments_per_row: num of segments per row,
      equal to num_qblocks_per_row/QBLOCKS_PER_SEGMENT.
      rows in qweight and x are loaded and computed segment by segment.
    num_invalid_qblocks_per_row: num of block_q8_0 that need to be padded at
      end of each row in qweight and x, to make last segment complete.
      Actually we don't pad the row but simply do no computation and pretend
      the dot product of invalid blocks as 0.


  Matmul kernel to perform quantized matrix multiplication:
      out = (qweight * qx.T).T

  If we temporarily ignore quantization, element in out is:
      out_i_j = sum(qweight_j_k * qx_i_k) for k in range(ncols),

  so for (i, j) in out, we need x(i, :) and qweight(j, :), in which ":" means all elements in that dimension.


  We perform matmul tilewise, one tile (of out) per block, tile size is defined as TILE_SIZE_COL and TILE_SIZE_ROW,
  so each block handle TILE_SIZE_COL * TILE_SIZE_ROW output elements.

  For computation in one block:
  Since we can't assume that all input data that required for output of this tile can be loaded into shared memory,
  so we load just one segment of the input data, do the partial matmul, and add result to a temperary sum.
  Do this until we done for all input data requierd, and we got the final output.
  A segment contains multiple quant blocks from the same row of input qweight or qx.


  Each block contains WARP_SIZE_GCN * TILE_SIZE_COL * TILE_SIZE_ROW threads,
  each thread handles one int8x4 dot product of one block, so 8 threads is needed to handle a single quant block dot product.

  So this kernel is expected to be launched with:
      gridDim = (ncols_out/TILE_SIZE_COL, nrows_out/TILE_SIZE_ROW)
      blockDim = (WARP_SIZE_GCN, TILE_SIZE_COL, TILE_SIZE_ROW)

  Thread indexing: since qblock in a segment is contigious in global memory,
  for coalescing memory access, thread indexing rule is:
      threadIdx.x -> int8x4 index in current segment (NOTE: current quant block idx in segment = threadIdx.x / 8)
      threadIdx.y -> column coordinate of element in tile output
      threadIdx.z -> row coordinate of element in tile output


  For each thread:

  Step 1: allocate shared memory and registers

  Step 2:
    for input_segment_index in all_segments:
      1. if current thread belongs to tile 1st row or 1st col:
        load input int8x4 to shared memory, if threadIdx.x % 8 == 0, also load qweight.d and x.d to register
      2. do vecdot with loaded int8x4 data
      3. reduce vecdot result to thread with idx.x % 8 == 0, and in that thread, multiply result with qweight.d and x.d
      now we've done one qblock's vecdot with 8 threads (64 threads in a warp -> 8 qblocks done in a warp)
      4. reduce qblock dot result in a warp to idx.x == 0, add to partial_sum

  Step 3: after the loop, we've done all vecdot over one row in qweight and x (all segments)
          we can reduce partial_sum to full output sum (of a single element) now.

  Step 4: write output back to global memory
  */

  int qblock_idx_in_segment = threadIdx.x / 8;
  int int8x4_idx_in_qblock = threadIdx.x % 8;
  int tile_col_idx = threadIdx.y;
  int tile_row_idx = threadIdx.z;

  int tile_col_idx_bias = TILE_SIZE_COL * blockIdx.x;
  int tile_row_idx_bias = TILE_SIZE_ROW * blockIdx.y;

  int global_out_col_idx = tile_col_idx_bias + tile_col_idx;
  int global_out_row_idx = tile_row_idx_bias + tile_row_idx;

  int &nrows_out = nrows_x;
  int &ncols_out = nrows_weight;

  if (global_out_row_idx >= nrows_out || global_out_col_idx >= ncols_out)
  {
    return;
  }

  // Step1: allocate shared memory and registers
  int32_t *smem_qweight_segment_qs = nullptr;
  half *smem_qweight_segment_d = nullptr;
  int32_t *smem_x_segment_qs = nullptr;
  half *smem_x_segment_d = nullptr;

  allocate_shared_memory(
      smem_qweight_segment_qs,
      smem_qweight_segment_d,
      smem_x_segment_qs,
      smem_x_segment_d);

  int32_t partial_sum_of_sdot4 = 0;
  float partial_sum_of_warp = 0.0f;
  float sum = 0.0f;

  // Step2: calculation
  // loop over all_segments
  for (int segment_idx = 0; segment_idx < num_segments_per_row; segment_idx++)
  {
    // load input data segment from global memory to shared memory
    int num_vald_qblock_in_segment = segment_idx != num_segments_per_row - 1 ? QBLOCKS_PER_SEGMENT
                                                                             : QBLOCKS_PER_SEGMENT - num_invald_qblocks_per_row;
    // printf("t(%d,%d,%d), num_vald_qblock_in_segment:%d\n",threadIdx.x, threadIdx.y, threadIdx.z, num_vald_qblock_in_segment);

    if (qblock_idx_in_segment < num_vald_qblock_in_segment)
    {
      // Only threads with tile_row_idx == 0 or tile_col_idx == 0 loads qblocks.
      // Thread only load one block of its segment of its row/col from qx/qweight.
      if (tile_col_idx == 0)
      {
        load_int8x4(qweight, smem_qweight_segment_qs,
                    global_out_col_idx, global_out_row_idx, num_qblocks_per_row,
                    tile_row_idx,
                    segment_idx, qblock_idx_in_segment, int8x4_idx_in_qblock);
        if (int8x4_idx_in_qblock == 0)
        {
          load_d(qweight, smem_qweight_segment_d,
                 global_out_col_idx, global_out_row_idx, num_qblocks_per_row,
                 tile_row_idx,
                 segment_idx, qblock_idx_in_segment);
        }
      }
      if (tile_row_idx == 0)
      {
        load_int8x4(qx, smem_x_segment_qs,
                    global_out_col_idx, global_out_row_idx, num_qblocks_per_row,
                    tile_col_idx,
                    segment_idx, qblock_idx_in_segment, int8x4_idx_in_qblock);
        if (int8x4_idx_in_qblock == 0)
        {
          load_d(qx, smem_x_segment_d,
                 global_out_col_idx, global_out_row_idx, num_qblocks_per_row,
                 tile_col_idx,
                 segment_idx, qblock_idx_in_segment);
        }
      }
    }
    __syncthreads();

    // do vecdot with loaded int8x4 data
    if (qblock_idx_in_segment < num_vald_qblock_in_segment)
    {
      int32_t a = smem_qweight_segment_qs[(tile_col_idx * QBLOCKS_PER_SEGMENT + qblock_idx_in_segment) * 8 + int8x4_idx_in_qblock];
      int32_t b = smem_x_segment_qs[(tile_row_idx * QBLOCKS_PER_SEGMENT + qblock_idx_in_segment) * 8 + int8x4_idx_in_qblock];
      // printf("thread(%d,%d,%d), a: %d, b: %d\n", threadIdx.x, threadIdx.y, threadIdx.z, a, b);
      partial_sum_of_sdot4 = __builtin_amdgcn_sdot4(a, b, partial_sum_of_sdot4, false);
    }
    else
    {
      partial_sum_of_sdot4 = 0;
    }
    // printf("thread(%d,%d,%d), partial_sum_of_sdot4: %d\n", threadIdx.x, threadIdx.y, threadIdx.z, partial_sum_of_sdot4);

    // reduce vecdot result to int8x4_idx_in_qblock=0 thread
#pragma unroll
    for (int delta = 4; delta > 0; delta /= 2)
    {
      partial_sum_of_sdot4 += __shfl_down_sync((uint64_t)-1, partial_sum_of_sdot4, delta, 64);
    }
    // now we have every qblock int dot result in int8x4_idx_in_qblock=0 threads
    if (int8x4_idx_in_qblock == 0)
    {
      // printf("thread(%d,%d,%d), partial_sum_of_sdot4(reduced): %d\n", threadIdx.x, threadIdx.y, threadIdx.z, partial_sum_of_sdot4);
    }

    // multiply d_x and d_qweight in int8x4_idx_in_qblock=0 thread
    if (int8x4_idx_in_qblock == 0)
    {
      float d_qweight = __half2float(smem_qweight_segment_d[tile_col_idx * QBLOCKS_PER_SEGMENT + qblock_idx_in_segment]);
      float d_x = __half2float(smem_x_segment_d[tile_row_idx * QBLOCKS_PER_SEGMENT + qblock_idx_in_segment]);
      partial_sum_of_warp = d_qweight * d_x * (float)partial_sum_of_sdot4;
      // printf("thread(%d,%d,%d), d1: %f, d2: %f, partial_sum_of_warp: %f\n", threadIdx.x, threadIdx.y, threadIdx.z, d_qweight, d_x, partial_sum_of_warp);
    }
    // now we have every qblock's final dot product result in partial_sum_of_warp in each int8x4_idx_in_qblock=0 thread

    // reduce partial_sum_of_warp to first thread of this warp
#pragma unroll
    for (int delta = 56; delta > 0; delta -= 8)
    {
      sum += __shfl_down_sync((uint64_t)-1, partial_sum_of_warp, delta, 64);
    }
    // now in every warp, we got final sum at 1st thread
    if (threadIdx.x == 0)
    {
      // printf("thread(%d,%d,%d), sum: %f\n", threadIdx.x, threadIdx.y, threadIdx.z, sum);
    }
  }

  // frist thread write output back to global memory
  if (threadIdx.x == 0 && global_out_row_idx < nrows_out && global_out_col_idx < ncols_out)
  {
    // printf("thread(%d,%d,%d): %f\n", threadIdx.x, threadIdx.y, threadIdx.z, sum);
    out[global_out_row_idx * nrows_weight + global_out_col_idx] = __float2half(sum);
  }
}