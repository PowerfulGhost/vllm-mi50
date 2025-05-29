#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "quantize_q8_0_mi50.cuh"

#include "debug_utils.cuh"

#define WARP_SIZE_GCN 64

// block size
#define BLOCK_SIZE_SEGMENT 8 // blockIdx.x direction
#define BLOCK_SIZE_COL 16     // blockIdx.y direction
#define BLOCK_SIZE_ROW 4     // blockIdx.z direction

// matmul tile size def
#define TILE_SIZE_COL 16 // unit: (unquantized) element
#define TILE_SIZE_ROW 16 //  unit: (unquantized) element

__device__ __forceinline__ void allocate_shared_memory(int32_t *&qweight_qs, half *&qweight_d,
                                                       int32_t *&x_qs, half *&x_d)
{
    __shared__ int32_t _qweight_qs[TILE_SIZE_COL * BLOCK_SIZE_SEGMENT * QI8_0];
    __shared__ int32_t _x_qs[TILE_SIZE_ROW * BLOCK_SIZE_SEGMENT * QI8_0];
    __shared__ half _qweight_d[TILE_SIZE_COL * BLOCK_SIZE_SEGMENT];
    __shared__ half _x_d[TILE_SIZE_ROW * BLOCK_SIZE_SEGMENT];

    // if (is_first_block() && is_first_thread())
    // {
    //     for (auto i = 0; i < TILE_SIZE_COL * BLOCK_SIZE_SEGMENT * QI8_0; i++)
    //     {
    //         _qweight_qs[i] = 0;
    //     }
    //     printf("qweght_qs inited\n");
    //     for (auto i = 0; i < TILE_SIZE_ROW * BLOCK_SIZE_SEGMENT * QI8_0; i++)
    //     {
    //         _x_qs[i] = 0;
    //     }
    //     printf("x_qs inited\n");
    //     for (auto i = 0; i < TILE_SIZE_COL * BLOCK_SIZE_SEGMENT; i++)
    //     {
    //         _qweight_d[i] = __float2half(0.0f);
    //     }
    //     printf("qweght_d inited\n");
    //     for (auto i = 0; i < TILE_SIZE_ROW * BLOCK_SIZE_SEGMENT; i++)
    //     {
    //         _x_d[i] = __float2half(0.0f);
    //     }
    //     printf("x_d inited\n");
    // }

    qweight_qs = _qweight_qs;
    qweight_d = _qweight_d;
    x_qs = _x_qs;
    x_d = _x_d;
}

// #define MAX_SMEM_QS_ADDR_BIAS (TILE_SIZE_COL * BLOCK_SIZE_SEGMENT * QI8_0 - 1)
// #define MAX_SMEM_D_ADDR_BIAS (TILE_SIZE_COL * BLOCK_SIZE_SEGMENT - 1)
__device__ __forceinline__ void load_qblock_to_shared_memory(block_q8_0 *qblock_start, int32_t *smem_qs_start, half *smem_d_start,
                                                             int global_col_or_row_idx, int tile_col_or_row_idx, int num_qblocks_per_row, int segment_idx, int segment_idx_in_smem,
                                                             int row_stride, int col_stride, bool loading_qweight)
{
    block_q8_0 *qblock = qblock_start + global_col_or_row_idx * num_qblocks_per_row + segment_idx;

    int _bias = tile_col_or_row_idx * BLOCK_SIZE_SEGMENT + segment_idx_in_smem;
    int32_t *qs = smem_qs_start + _bias * QI8_0;
    half *d = smem_d_start + _bias;

    // if (loading_qweight)
    //     printf("t%d,%d,%d, row_stride:%d, col_stride:%d, w | qblock_addr_bias:%d, smem_qs_bias:%d(max:%d), smem_d_bias:%d(max:%d)\n", threadIdx.x, threadIdx.y, threadIdx.z, row_stride, col_stride, qblock_addr_bias, smem_qs_bias, MAX_SMEM_QS_ADDR_BIAS, smem_d_bias, MAX_SMEM_D_ADDR_BIAS);
    // else
    //     printf("t%d,%d,%d, row_stride:%d, col_stride:%d, x | qblock_addr_bias:%d, smem_qs_bias:%d(max:%d), smem_d_bias:%d(max:%d)\n", threadIdx.x, threadIdx.y, threadIdx.z, row_stride, col_stride, qblock_addr_bias, smem_qs_bias, MAX_SMEM_QS_ADDR_BIAS, smem_d_bias, MAX_SMEM_D_ADDR_BIAS);

    memcpy(qs, qblock->qs, QI8_0 * sizeof(int32_t));
    *d = qblock->d;
}

// __device__ __forceinline__ void load_qblock_to_shared_memory_dbg(block_q8_0 *qblock_start, int32_t *smem_qs_start, half *smem_d_start,
//                                                                  int global_col_or_row_idx, int tile_col_or_row_idx, int num_qblocks_per_row, int segment_idx, int segment_idx_in_smem)
// {
//     block_q8_0 *qblock = qblock_start + global_col_or_row_idx * num_qblocks_per_row + segment_idx;

//     if (is_first_block() && is_first_thread())
//     {
//         print_qblock(qblock, 1);
//     }

//     // printf("b%d,%d,%d,t%d,%d,%d, qblock_start:%p, qblock:%p, global_col_or_row_idx:%d, segment_idx:%d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, (void *)qblock_start, (void *)qblock, global_col_or_row_idx, segment_idx);

//     int32_t *qs = smem_qs_start + (tile_col_or_row_idx * BLOCK_SIZE_SEGMENT + segment_idx_in_smem) * QI8_0;

//     half *d = smem_d_start + tile_col_or_row_idx * BLOCK_SIZE_SEGMENT + segment_idx_in_smem;

//     memcpy(qs, qblock->qs, QI8_0 * sizeof(int32_t));
//     *d = qblock->d;
// }

__device__ __forceinline__ void get_qblock_from_shared_memory(int32_t *qs_start, half *d_start, block_q8_0 *out,
                                                              int tile_col_or_row_idx, int segment_idx_in_smem)
{
    int32_t *qs = qs_start + (tile_col_or_row_idx * BLOCK_SIZE_SEGMENT + segment_idx_in_smem) * QI8_0;
    half *d = d_start + tile_col_or_row_idx * BLOCK_SIZE_SEGMENT + segment_idx_in_smem;

    memcpy(out->qs, qs, QI8_0 * sizeof(int32_t));
    out->d = *d;
}

__device__ __forceinline__ float vec_dot_q8_0_q8_0(block_q8_0 *b0,
                                                   block_q8_0 *b1)
{
    /*
    result = d0 * d1 * (qs_a_0*qs_1_0 + ... + qs_0_31*qs_1_31)
    in which d0 and d1 are scale factor (float16), qs_x_i is quantized weight (int8).

    we do int8 dot product using __builtin_amdgcn_sdot4, other keywords: llvm.amdgcn.sdot4, v_dot4_i32_i8

    references:
    https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Basic/BuiltinsAMDGPU.def?spm=a2ty_o01.29997173.0.0.2f5ec921WpP3rh&file=BuiltinsAMDGPU.def
    https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html
    https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/vega-7nm-shader-instruction-set-architecture.pdf
    */

    float d_q8_0 = __half2float(b0->d);
    float d_q8_1 = __half2float(b1->d);

    int32_t int_product_result = 0;
    int32_t inner_product_out = 0;

#pragma unroll
    for (int i = 0; i < QK8_0; i += 4)
    {
        int32_t *a = reinterpret_cast<int32_t *>(b0->qs + i);
        int32_t *b = reinterpret_cast<int32_t *>(b1->qs + i);
        int_product_result =
            __builtin_amdgcn_sdot4(*a, *b, int_product_result, false);
    }

    return d_q8_0 * d_q8_1 * (float)int_product_result;
}

__global__ void mul_mat_q8_0_mi50(block_q8_0 *__restrict__ qweight,
                                  block_q8_0 *__restrict__ qx,
                                  half *__restrict__ out, int ncols,
                                  int nrows_weight, int nrows_x,
                                  int num_qblocks_per_row)
{

    // if (is_first_block() && is_first_thread())
    // {
    //     printf("HBM: qweight:\n");
    //     print_qblock(qweight, nrows_weight * ncols / QK8_0);
    //     printf("HBM: qx:\n");
    //     print_qblock(qx, nrows_x * ncols / QK8_0);
    // }

    auto tile_col_idx_bias = TILE_SIZE_COL * blockIdx.x;
    auto tile_row_idx_bias = TILE_SIZE_ROW * blockIdx.y;

    auto &nrows_out = nrows_x;
    auto &ncols_out = nrows_weight;

    // Allocate shared memory and registers
    int32_t *smem_qweight_qs = nullptr;
    half *smem_qweight_d = nullptr;
    int32_t *smem_x_qs = nullptr;
    half *smem_x_d = nullptr;

    allocate_shared_memory(smem_qweight_qs, smem_qweight_d, smem_x_qs, smem_x_d);

    // if (is_first_block() && is_first_thread())
    // {
    //     printf("BEFORE LOOP START\n");
    //     printf("smem_qweight after alloc:\n");
    //     print_smem(smem_qweight_qs, smem_qweight_d, TILE_SIZE_COL * BLOCK_SIZE_SEGMENT);
    //     printf("smem_x after alloc:\n");
    //     print_smem(smem_x_qs, smem_x_d, TILE_SIZE_ROW * BLOCK_SIZE_SEGMENT);
    // }

    float sum[TILE_SIZE_ROW / BLOCK_SIZE_ROW][TILE_SIZE_COL / BLOCK_SIZE_COL] = {{0.0f}};
    // __syncthreads();

    // Calculation
    // loop over segments
#pragma unroll
    for (auto segment_start_idx = 0; segment_start_idx < num_qblocks_per_row; segment_start_idx += BLOCK_SIZE_SEGMENT)
    {
#pragma unroll
        for (auto row_stride = 0; row_stride < TILE_SIZE_ROW / BLOCK_SIZE_ROW; row_stride++)
        {
#pragma unroll
            for (auto col_stride = 0; col_stride < TILE_SIZE_COL / BLOCK_SIZE_COL; col_stride++)
            {
                auto segment_idx = segment_start_idx + threadIdx.x;
                auto tile_col_idx = BLOCK_SIZE_COL * col_stride + threadIdx.y;
                auto tile_row_idx = BLOCK_SIZE_ROW * row_stride + threadIdx.z;

                auto global_out_col_idx = tile_col_idx_bias + tile_col_idx;
                auto global_out_row_idx = tile_row_idx_bias + tile_row_idx;

                if (global_out_col_idx >= ncols_out || global_out_row_idx >= nrows_out || segment_idx >= num_qblocks_per_row)
                    continue;

                // if (is_first_block() && is_first_thread())
                // {
                //     printf("segment_start_idx:%d, row_stride:%d, col_stride:%d\n", segment_start_idx, row_stride, col_stride);
                //     printf("BEFORE_LOAD\n");
                //     printf("smem_qweight:\n");
                //     print_smem(smem_qweight_qs, smem_qweight_d, TILE_SIZE_COL * BLOCK_SIZE_SEGMENT);
                //     printf("smem_x:\n");
                //     print_smem(smem_x_qs, smem_x_d, TILE_SIZE_ROW * BLOCK_SIZE_SEGMENT);
                // }

                // __syncthreads();
                // load input data segment from global memory to shared memory
                if (tile_row_idx == 0)
                {
                    load_qblock_to_shared_memory(qweight, smem_qweight_qs, smem_qweight_d, global_out_col_idx, tile_col_idx, num_qblocks_per_row, segment_idx, threadIdx.x, row_stride, col_stride, true);
                }
                if (tile_col_idx == 0)
                {
                    load_qblock_to_shared_memory(qx, smem_x_qs, smem_x_d, global_out_row_idx, tile_row_idx, num_qblocks_per_row, segment_idx, threadIdx.x, row_stride, col_stride, false);
                }
                __syncthreads();

                // if (is_first_block() && is_first_thread())
                // {
                //     printf("segment_start_idx:%d, row_stride:%d, col_stride:%d\n", segment_start_idx, row_stride, col_stride);
                //     //     printf("AFTER_LOAD\n");
                //     //     printf("smem_qweight:\n");
                //     //     print_smem(smem_qweight_qs, smem_qweight_d, TILE_SIZE_COL * BLOCK_SIZE_SEGMENT);
                //     //     printf("smem_x:\n");
                //     //     print_smem(smem_x_qs, smem_x_d, TILE_SIZE_ROW * BLOCK_SIZE_SEGMENT);
                // }

                // do vecdot for the qblock of this thread
                block_q8_0 qblock_qweight, qblock_x;

                get_qblock_from_shared_memory(smem_x_qs, smem_x_d, &qblock_x, tile_row_idx, threadIdx.x);
                get_qblock_from_shared_memory(smem_qweight_qs, smem_qweight_d, &qblock_qweight, tile_col_idx, threadIdx.x);

                // print_single_qblock(qblock_qweight);
                // print_single_qblock(qblock_x);

                sum[row_stride][col_stride] += vec_dot_q8_0_q8_0(&qblock_qweight, &qblock_x);
                // __syncthreads();
            }
        }
    }

    // if (is_first_block() && is_first_thread())
    // {
    //     // printf("segment_start_idx:%d, row_stride:%d, col_stride:%d\n", segment_start_idx, row_stride, col_stride);
    //     //     printf("AFTER_LOAD\n");
    //     printf("smem_qweight:\n");
    //     print_smem(smem_qweight_qs, smem_qweight_d, TILE_SIZE_COL * BLOCK_SIZE_SEGMENT);
    //     printf("smem_x:\n");
    //     print_smem(smem_x_qs, smem_x_d, TILE_SIZE_ROW * BLOCK_SIZE_SEGMENT);
    // }

    // printf("BEFORE SHFL | b%d,%d,%d,t%d,%d,%d | sum: %f, %f, %f, %f\n",
    //        blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,
    //        sum[0][0], sum[0][1], sum[1][0], sum[1][1]);

    // shuffle sum to threadIdx.x=0
#pragma unroll
    for (auto delta = 2; delta > 0; delta /= 2)
    {
#pragma unroll
        for (auto row_stride = 0; row_stride < TILE_SIZE_ROW / BLOCK_SIZE_ROW; row_stride++)
        {
#pragma unroll
            for (auto col_stride = 0; col_stride < TILE_SIZE_COL / BLOCK_SIZE_COL; col_stride++)
            {
                sum[row_stride][col_stride] += __shfl_down_sync((uint64_t)-1, sum[row_stride][col_stride], delta);
            }
        }
    }

    // printf("AFTER SHFL | b%d,%d,%d,t%d,%d,%d | sum: %f, %f, %f, %f\n",
    //        blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,
    //        sum[0][0], sum[0][1], sum[1][0], sum[1][1]);

    // write output back to global memory
#pragma unroll
    for (auto row_stride = 0; row_stride < TILE_SIZE_ROW / BLOCK_SIZE_ROW; row_stride++)
    {
#pragma unroll
        for (auto col_stride = 0; col_stride < TILE_SIZE_COL / BLOCK_SIZE_COL; col_stride++)
        {
            auto tile_col_idx = BLOCK_SIZE_COL * col_stride + threadIdx.y;
            auto tile_row_idx = BLOCK_SIZE_ROW * row_stride + threadIdx.z;

            auto global_out_col_idx = tile_col_idx_bias + tile_col_idx;
            auto global_out_row_idx = tile_row_idx_bias + tile_row_idx;

            if (global_out_col_idx >= ncols_out || global_out_row_idx >= nrows_out || threadIdx.x != 0)
                continue;

            out[global_out_row_idx * ncols_out + global_out_col_idx] = sum[row_stride][col_stride];
        }
    }
}

torch::Tensor ggml_mul_mat_a8_q8_0_mi50(torch::Tensor W, // quant weight, q8_0
                                        torch::Tensor X  // input, fp16
)
{
    int ncols = X.size(1);
    if (ncols % 32 != 0)
    {
        throw std::invalid_argument("x.shape[1] must be multiple of 32!\n");
    }
    int nrows_x = X.size(0);
    int nrows_w = W.size(0);

    int block_q8_0_per_row = (ncols + QK8_0 - 1) / QK8_0;
    const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // quantized X
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(W.device());
    at::Tensor quant_X = torch::empty({nrows_x, block_q8_0_per_row * (int)sizeof(block_q8_0)}, options);

    const dim3 gridDim_quant((ncols + WARP_SIZE_GCN - 1) / WARP_SIZE_GCN, nrows_x);
    const dim3 blockDim_quant(WARP_SIZE_GCN);

    // printf("INPUT_X:\n");
    // print_float_tensor(X);
    quantize_q8_0_mi50<<<gridDim_quant, blockDim_quant, 0, stream>>>((half *)X.data_ptr(), (block_q8_0 *)quant_X.data_ptr(), nrows_x, ncols, block_q8_0_per_row);

    // printf("RIGHT AFTER X QUANTIZATION:\n");
    // print_qblock((block_q8_0 *)quant_X.data_ptr(), nrows_x * ncols / QK8_0);

    // do matmul
    int &nrows_out = nrows_x;
    int &ncols_out = nrows_w;

    options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
    at::Tensor Y = torch::zeros({nrows_out, ncols_out}, options);

    int num_qblocks_per_row = (ncols + QK8_0 - 1) / QK8_0;

    int num_tiles_col = (ncols_out + TILE_SIZE_COL - 1) / TILE_SIZE_COL;
    int num_tiles_row = (nrows_out + TILE_SIZE_ROW - 1) / TILE_SIZE_ROW;
    const dim3 gridDim_mulmat(num_tiles_col, num_tiles_row);

    const dim3 blockDim_mulmat(BLOCK_SIZE_SEGMENT, BLOCK_SIZE_COL, BLOCK_SIZE_ROW); // for v50
    // const dim3 blockDim_mulmat(BLOCK_SIZE_ROW, BLOCK_SIZE_COL, BLOCK_SIZE_SEGMENT); // for v46

    mul_mat_q8_0_mi50<<<gridDim_mulmat, blockDim_mulmat, 0, stream>>>(
        (block_q8_0 *)W.data_ptr(), (block_q8_0 *)quant_X.data_ptr(),
        (half *)Y.data_ptr(), ncols, nrows_w, nrows_x, num_qblocks_per_row);

    return Y;
}