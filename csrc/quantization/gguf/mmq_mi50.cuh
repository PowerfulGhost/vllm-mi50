#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "quantize_q8_0_mi50.cuh"

#include "mmq_mi50_best_size_lookup.h"

#define WARP_SIZE_GCN 64

// reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared
extern __shared__ char smem[];
__device__ __forceinline__ void allocate_shared_memory(int32_t *&qw_qs, half *&qw_d, int32_t *&x_qs, half *&x_d,
                                                       int &tile_size_col, int &tile_size_row, int &block_size_segment)
{
    const int qw_qs_size = tile_size_col * block_size_segment * QI8_0 * sizeof(int32_t);
    const int x_qs_size = tile_size_row * block_size_segment * QI8_0 * sizeof(int32_t);
    const int qw_d_size = tile_size_col * block_size_segment * sizeof(half);

    qw_qs = (int32_t *)smem;
    x_qs = (int32_t *)&smem[qw_qs_size];
    qw_d = (half *)&smem[qw_qs_size + x_qs_size];
    x_d = (half *)&smem[qw_qs_size + x_qs_size + qw_d_size];
}

__device__ __forceinline__ void load_qblock_to_shared_memory(block_q8_0 *qblock_start, int32_t *smem_qs_start, half *smem_d_start,
                                                             int global_qblock_idx, int smem_qblock_idx)
{
    block_q8_0 *qblock = qblock_start + global_qblock_idx;
    int32_t *qs = smem_qs_start + smem_qblock_idx * QI8_0;
    half *d = smem_d_start + smem_qblock_idx;

    memcpy(qs, qblock->qs, QI8_0 * sizeof(int32_t));
    *d = qblock->d;
}

__device__ __forceinline__ void get_qblock_from_shared_memory(int32_t *qs_start, half *d_start, block_q8_0 *out, int qblock_idx)
{
    int32_t *qs = qs_start + qblock_idx * QI8_0;
    half *d = d_start + qblock_idx;

    memcpy(out->qs, qs, QI8_0 * sizeof(int32_t));
    out->d = *d;
}

__device__ __forceinline__ float vec_dot_q8_0_q8_0(block_q8_0 *b0,
                                                   block_q8_0 *b1)
{
    /*
    result = d0 * d1 * (qs_a_0*qs_1_0 + ... + qs_0_31*qs_1_31)
    in which d0 and d1 are scale factor (float16), qs_x_i is quantized weight (int8).

    int8 dot product using __builtin_amdgcn_sdot4, other keywords: llvm.amdgcn.sdot4, v_dot4_i32_i8
    references:
    https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Basic/BuiltinsAMDGPU.def?spm=a2ty_o01.29997173.0.0.2f5ec921WpP3rh&file=BuiltinsAMDGPU.def
    https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html
    https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/vega-7nm-shader-instruction-set-architecture.pdf
    */

    float d_q8_0 = __half2float(b0->d);
    float d_q8_1 = __half2float(b1->d);

    int32_t int_product_result = 0;

#pragma unroll
    for (int i = 0; i < QK8_0; i += 4)
    {
        int32_t *a = reinterpret_cast<int32_t *>(b0->qs + i);
        int32_t *b = reinterpret_cast<int32_t *>(b1->qs + i);
        int_product_result = __builtin_amdgcn_sdot4(*a, *b, int_product_result, false);
    }

    return d_q8_0 * d_q8_1 * (float)int_product_result;
}

template <int SUBTILE_SIZE_COL, int SUBTILE_SIZE_ROW>
__global__ void mul_mat_q8_0_mi50(block_q8_0 *__restrict__ qw,
                                  block_q8_0 *__restrict__ qx,
                                  half *__restrict__ out,
                                  int nrows_w, int nrows_x, int num_qblocks_per_row,
                                  int tile_size_col, int tile_size_row,
                                  int block_size_segment, int block_size_col, int block_size_row)
{
    auto tile_col_idx_bias = tile_size_col * blockIdx.x;
    auto tile_row_idx_bias = tile_size_row * blockIdx.y;

    auto &nrows_out = nrows_x;
    auto &ncols_out = nrows_w;

    // Allocate shared memory and registers
    int32_t *smem_qw_qs = nullptr;
    half *smem_qw_d = nullptr;
    int32_t *smem_qx_qs = nullptr;
    half *smem_qx_d = nullptr;

    allocate_shared_memory(smem_qw_qs, smem_qw_d, smem_qx_qs, smem_qx_d, tile_size_col, tile_size_row, block_size_segment);

    float sum[SUBTILE_SIZE_ROW][SUBTILE_SIZE_COL] = {{0.0f}};

    // Calculation
    for (auto segment_stride = 0; segment_stride <= num_qblocks_per_row / block_size_segment; segment_stride++)
    {
        for (auto row_stride = 0; row_stride < SUBTILE_SIZE_ROW; row_stride++)
        {
            for (auto col_stride = 0; col_stride < SUBTILE_SIZE_COL; col_stride++)
            {
                auto segment_idx = block_size_segment * segment_stride + threadIdx.x;
                auto tile_col_idx = block_size_col * col_stride + threadIdx.y;
                auto tile_row_idx = block_size_row * row_stride + threadIdx.z;
                auto global_col_idx = tile_col_idx_bias + tile_col_idx;
                auto global_row_idx = tile_row_idx_bias + tile_row_idx;

                if (global_col_idx < ncols_out && global_row_idx < nrows_out && segment_idx < num_qblocks_per_row)
                {
                    // load input data segment from global memory to shared memory
                    if (tile_row_idx == 0)
                    {
                        int global_qblock_idx = global_col_idx * num_qblocks_per_row + segment_idx;
                        int smem_qblock_idx = tile_col_idx * block_size_segment + threadIdx.x;
                        load_qblock_to_shared_memory(qw, smem_qw_qs, smem_qw_d, global_qblock_idx, smem_qblock_idx);
                    }
                    if (tile_col_idx == 0)
                    {
                        int global_qblock_idx = global_row_idx * num_qblocks_per_row + segment_idx;
                        int smem_qblock_idx = tile_row_idx * block_size_segment + threadIdx.x;
                        load_qblock_to_shared_memory(qx, smem_qx_qs, smem_qx_d, global_qblock_idx, smem_qblock_idx);
                    }
                }

                __syncthreads();

                if (global_col_idx < ncols_out && global_row_idx < nrows_out && segment_idx < num_qblocks_per_row)
                {
                    // do vecdot for the qblock of this thread
                    block_q8_0 qblock_qw, qblock_x;

                    int qblock_idx_qw = tile_col_idx * block_size_segment + threadIdx.x;
                    int qblock_idx_qx = tile_row_idx * block_size_segment + threadIdx.x;

                    get_qblock_from_shared_memory(smem_qw_qs, smem_qw_d, &qblock_qw, qblock_idx_qw);
                    get_qblock_from_shared_memory(smem_qx_qs, smem_qx_d, &qblock_x, qblock_idx_qx);

                    float _sum = vec_dot_q8_0_q8_0(&qblock_qw, &qblock_x);

                    sum[row_stride][col_stride] += _sum;
                }

                __syncthreads();
            }
        }
    }

    // shuffle sum to threadIdx.x=0
    for (auto delta = block_size_segment / 2; delta > 0; delta /= 2)
    {
        for (auto row_stride = 0; row_stride < SUBTILE_SIZE_ROW; row_stride++)
        {
            for (auto col_stride = 0; col_stride < SUBTILE_SIZE_COL; col_stride++)
            {
                sum[row_stride][col_stride] += __shfl_down_sync((uint64_t)-1, sum[row_stride][col_stride], delta);
            }
        }
    }

    // write output back to global memory
    for (auto row_stride = 0; row_stride < SUBTILE_SIZE_ROW; row_stride++)
    {
        for (auto col_stride = 0; col_stride < SUBTILE_SIZE_COL; col_stride++)
        {
            auto tile_col_idx = block_size_col * col_stride + threadIdx.y;
            auto tile_row_idx = block_size_row * row_stride + threadIdx.z;

            auto global_col_idx = tile_col_idx_bias + tile_col_idx;
            auto global_row_idx = tile_row_idx_bias + tile_row_idx;

            if (global_col_idx < ncols_out && global_row_idx < nrows_out && threadIdx.x == 0)
            {
                out[global_row_idx * ncols_out + global_col_idx] = sum[row_stride][col_stride];
            }
        }
    }
}

#define _LAUNCH_MI50_GGUF_MULMAT_KERNEL(SUBTILE_SIZE_COL, SUBTILE_SIZE_ROW)                                                       \
    do                                                                                                                       \
    {                                                                                                                        \
        mul_mat_q8_0_mi50<SUBTILE_SIZE_COL, SUBTILE_SIZE_ROW><<<gridDim_mulmat, blockDim_mulmat, smem_size_bytes, stream>>>( \
            (block_q8_0 *)W.data_ptr(), (block_q8_0 *)quant_X.data_ptr(), (half *)Y.data_ptr(),                              \
            nrows_w, nrows_x, num_qblocks_per_row,                                                                           \
            tile_size_col, tile_size_row,                                                                                    \
            block_size_segment, block_size_col, block_size_row);                                                             \
    } while (0)

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

    quantize_q8_0_mi50<<<gridDim_quant, blockDim_quant, 0, stream>>>((half *)X.data_ptr(), (block_q8_0 *)quant_X.data_ptr(), nrows_x, ncols, block_q8_0_per_row);

    // do matmul
    int &nrows_out = nrows_x;
    int &ncols_out = nrows_w;

    options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
    at::Tensor Y = torch::empty({nrows_out, ncols_out}, options);

    int num_qblocks_per_row = ncols / QK8_0; // ncols % 32 == 0 is guaranteed

    auto sizes = size_lookup(nrows_w, nrows_x, ncols);
    int &tile_size_col = sizes[0];
    int &tile_size_row = sizes[1];
    int &block_size_segment = sizes[2];
    int &block_size_col = sizes[3];
    int &block_size_row = sizes[4];

    int num_tiles_col = (ncols_out + tile_size_col - 1) / tile_size_col;
    int num_tiles_row = (nrows_out + tile_size_row - 1) / tile_size_row;
    const dim3 gridDim_mulmat(num_tiles_col, num_tiles_row);
    const dim3 blockDim_mulmat(block_size_segment, block_size_col, block_size_row);

    int smem_size_bytes = (QI8_0 * sizeof(int32_t) + sizeof(half)) * block_size_segment * (tile_size_col + tile_size_row);

    int subtile_size_col = tile_size_col / block_size_col;
    int subtile_size_row = tile_size_row / block_size_row;
    int _key = subtile_size_col * 10 + subtile_size_row;

    switch (_key)
    {
    case 11:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(1, 1);
        break;
    case 12:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(1, 2);
        break;
    case 13:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(1, 3);
        break;
    case 14:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(1, 4);
        break;
    case 15:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(1, 5);
        break;
    case 16:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(1, 6);
        break;
    case 21:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(2, 1);
        break;
    case 22:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(2, 2);
        break;
    case 23:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(2, 3);
        break;
    case 24:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(2, 4);
        break;
    case 25:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(2, 5);
        break;
    case 26:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(2, 6);
        break;
    case 31:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(3, 1);
        break;
    case 32:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(3, 2);
        break;
    case 33:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(3, 3);
        break;
    case 34:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(3, 4);
        break;
    case 35:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(3, 5);
        break;
    case 36:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(3, 6);
        break;
    case 41:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(4, 1);
        break;
    case 42:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(4, 2);
        break;
    case 43:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(4, 3);
        break;
    case 44:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(4, 4);
        break;
    case 45:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(4, 5);
        break;
    case 46:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(4, 6);
        break;
    case 51:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(5, 1);
        break;
    case 52:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(5, 2);
        break;
    case 53:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(5, 3);
        break;
    case 54:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(5, 4);
        break;
    case 55:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(5, 5);
        break;
    case 56:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(5, 6);
        break;
    case 61:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(6, 1);
        break;
    case 62:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(6, 2);
        break;
    case 63:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(6, 3);
        break;
    case 64:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(6, 4);
        break;
    case 65:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(6, 5);
        break;
    case 66:
        _LAUNCH_MI50_GGUF_MULMAT_KERNEL(6, 6);
        break;
    }

    return Y;
}
