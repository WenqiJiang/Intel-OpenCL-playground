// Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

#include "constants.hpp"

 // ACL kernel for adding two input vectors
// restrict -> for pointer to an object, gurantee there's no change to the pointer
// this allows the compiler for better optimizations

#pragma OPENCL EXTENSION cl_intel_channels : enable

channel int embedding_vec_0 __attribute__((depth(FIFO_BATCH * INPUT_SIZE)));
// channel int embedding_vec_1 __attribute__((depth(FIFO_BATCH * INPUT_SIZE)));

// // Top-Level
// __kernel void vector_add(__global const D_TYPE* /*restrict*/ table_DRAM, 
//                          __global D_TYPE* /*restrict*/ output_DRAM)
// {

//     int output_local_0[BATCH_SIZE];
//     int output_local_1[BATCH_SIZE];

//     embedding_lookup(
//     __global const D_TYPE* /*restrict*/ table_DRAM
//     write_only pipe int __attribute__((depth(FIFO_BATCH * INPUT_SIZE))) singie_embedding_vec);

//     // compute
//     for (int item = 0; item < BATCH_SIZE; item++) {

//         int local_start_idx = item * INPUT_SIZE;
//         D_TYPE result = 0;
//         #pragma unroll 4
//         for (int count = 0; count < INPUT_SIZE; count++) {
//             result += embedding_local[local_start_idx + count];
//         }
//         output_local[item] = result;
//     }

//     // write back
//     for (int item = 0; item < BATCH_SIZE; item++) {
//         output_DRAM[item] = output_local[item];
//     }
// }

// __kernel void embedding_lookup(
//     __global const D_TYPE* restrict table_DRAM) {

//     const int access_idx[] = {3, 99, 38, 72, 29, 57, 1, 72, 36, 76, 35, 50, 37, 57, 
//         13, 66, 26, 70, 41, 93, 48, 82, 44, 78, 25, 52, 3, 92, 36, 56, 46, 88};

//     int embedding_local[BATCH_SIZE * INPUT_SIZE];

//     // load 
//     for (int batch = 0; batch < BATCH_NUM; batch++) {
//         for (int item = 0; item < BATCH_SIZE; item++) {

//             int idx = access_idx[item];
//             int local_start_idx = item * INPUT_SIZE;
//             // 3 tables
//             for (int count = 0; count < DATA_SIZE_0; count++) {
//                 embedding_local[VECTOR_START_0 + local_start_idx + count] = 
//                     table_DRAM[ADDR_START_TABLE_0 + idx * DATA_SIZE_0 + count];
//             }
//             for (int count = 0; count < DATA_SIZE_1; count++) {
//                 embedding_local[VECTOR_START_1 + local_start_idx + count] = 
//                     table_DRAM[ADDR_START_TABLE_1 + idx * DATA_SIZE_1 + count];
//             }
//             for (int count = 0; count < DATA_SIZE_2; count++) {
//                 embedding_local[VECTOR_START_2 + local_start_idx + count] = 
//                     table_DRAM[ADDR_START_TABLE_2 + idx * DATA_SIZE_2 + count];
//             }
//         }
//     }

//     for (int i = 0; i < BATCH_SIZE * INPUT_SIZE; i++) { 
//         int reg = embedding_local[i];
//         write_channel_intel(embedding_vec_0, reg);
//         // write_channel_intel (embedding_vec_1, reg);
//         // write_pipe(singie_embedding_vec, &embedding_local[i]); // use pointer
//     } 
// }


// __kernel void reduction_sum(
//     __global D_TYPE* /*restrict*/ output_DRAM) {

//     int output_local[BATCH_SIZE];
//     int embedding_single_item[INPUT_SIZE];
    
//     // compute
//     for (int item = 0; item < BATCH_SIZE; item++) {

//         for (int count = 0; count < INPUT_SIZE; count++) {
//             embedding_single_item[count] = 
//                 read_channel_intel(embedding_vec_0);
//         }

//         D_TYPE result = 0;
//         #pragma unroll 4
//         for (int count = 0; count < INPUT_SIZE; count++) {
//             result += embedding_single_item[count];
//         }
//         output_local[item] = result;
//     }

//     // write back
//     for (int item = 0; item < BATCH_SIZE; item++) {
//         output_DRAM[item] = output_local[item];
//     }
// }

__attribute__((reqd_work_group_size(1,1,1)))
__kernel void embedding_lookup(__global const D_TYPE* /*restrict*/ table_DRAM, 
                         __global D_TYPE* /*restrict*/ output_DRAM)
{
    // get index of the work item
    // int index = get_global_id(0);

    // add the vector elements
    // z[index] = x[index] + y[index];

    const int access_idx[] = {3, 99, 38, 72, 29, 57, 1, 72, 36, 76, 35, 50, 37, 57, 
      13, 66, 26, 70, 41, 93, 48, 82, 44, 78, 25, 52, 3, 92, 36, 56, 46, 88};

    int output_local[BATCH_SIZE];
    int embedding_local[BATCH_SIZE * INPUT_SIZE];

    // load 
    for (int batch = 0; batch < BATCH_NUM; batch++) {
        for (int item = 0; item < BATCH_SIZE; item++) {

            int idx = access_idx[item];
            int local_start_idx = item * INPUT_SIZE;
            // 3 tables
            for (int count = 0; count < DATA_SIZE_0; count++) {
                embedding_local[VECTOR_START_0 + local_start_idx + count] = 
                    table_DRAM[ADDR_START_TABLE_0 + idx * DATA_SIZE_0 + count];
            }
            for (int count = 0; count < DATA_SIZE_1; count++) {
                embedding_local[VECTOR_START_1 + local_start_idx + count] = 
                    table_DRAM[ADDR_START_TABLE_1 + idx * DATA_SIZE_1 + count];
            }
            for (int count = 0; count < DATA_SIZE_2; count++) {
                embedding_local[VECTOR_START_2 + local_start_idx + count] = 
                    table_DRAM[ADDR_START_TABLE_2 + idx * DATA_SIZE_2 + count];
            }
        }
    }

    // compute
    for (int item = 0; item < BATCH_SIZE; item++) {

        int local_start_idx = item * INPUT_SIZE;
        D_TYPE result = 0;
        #pragma unroll 4
        for (int count = 0; count < INPUT_SIZE; count++) {
            result += embedding_local[local_start_idx + count];
        }
        output_local[item] = result;
    }

    // write back
    for (int item = 0; item < BATCH_SIZE; item++) {
        output_DRAM[item] = output_local[item];
    }
}

