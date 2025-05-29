#pragma once

#include "ggml-common.h"
#include <stdio.h>
#include <cuda_fp16.h>
#include <torch/all.h>

void print_float_tensor(at::Tensor t)
{
    auto m = t.size(0);
    auto n = t.size(1);
    for (auto i = 0; i < m; i++)
    {
        for (auto j = 0; j < n; j++)
        {
            printf("%f,", t[i][j].item().toFloat());
        }
        printf("\n");
    }
}

__device__ __host__ void print_qblock(block_q8_0 *qblock, int n)
{
    for (int i = 0; i < n; i++)
    {
        block_q8_0 *b = qblock + i;
        printf("block %d:\nqs: ", i);
        for (int j = 0; j < QK8_0; j++)
        {
            printf("%d, ", b->qs[j]);
        }
        printf("\nd: %f\n", __half2float(b->d));
    }
}

__device__ void print_single_qblock(block_q8_0 qblock)
{
    printf("b%d,%d,%d,t%d,%d,%d | qs:%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d | d:%f\n",
           blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,
           qblock.qs[0], qblock.qs[1], qblock.qs[2], qblock.qs[3], qblock.qs[4], qblock.qs[5], qblock.qs[6], qblock.qs[7],
           qblock.qs[8], qblock.qs[9], qblock.qs[10], qblock.qs[11], qblock.qs[12], qblock.qs[13], qblock.qs[14], qblock.qs[15],
           qblock.qs[16], qblock.qs[17], qblock.qs[18], qblock.qs[19], qblock.qs[20], qblock.qs[21], qblock.qs[22], qblock.qs[23],
           qblock.qs[24], qblock.qs[25], qblock.qs[26], qblock.qs[27], qblock.qs[28], qblock.qs[29], qblock.qs[30], qblock.qs[31],
           __half2float(qblock.d));
}

__device__ __host__ void print_smem(int32_t *smem_qs, half *smem_d, int num_qblocks)
{
    int8_t *smem_qs_int8 = (int8_t *)smem_qs;
    for (int i = 0; i < num_qblocks; i++)
    {
        printf("block %d:\n", i);
        printf("qs: ");
        for (int j = 0; j < QK8_0; j++)
        {
            printf("%d, ", smem_qs_int8[i * QK8_0 + j]);
        }
        printf("\nd: %f\n", __half2float(smem_d[i]));
    }
}

__device__ bool is_first_block()
{
    return blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
}

__device__ bool is_first_thread()
{
    return threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
}
