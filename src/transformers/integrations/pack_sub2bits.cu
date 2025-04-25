#include <cuda_bf16.h> 
#include <stdio.h>
#include <math.h>

typedef unsigned char uint8_t;

extern "C"
__global__ void pack_sub2bits(
	const __nv_bfloat16 *__restrict__ src,
	uint8_t 		*__restrict__ dst,
	int nrows,
	int ncols,
	int packed_ncols
) {
	
    /* 
	* Dimensions:
    * Each block handles one row. 
    * Each block has `packed_ncols` number of threads. So each thread handles 
    * one packing group, i.e. 5 values into 1 uint8. 
	*/

	int row_id	= blockIdx.x;
	int tid 	= threadIdx.x;
	int block_size = blockDim.x; // n_threads in the block

	const unsigned int pow[5] = {1, 3, 9, 27, 81};

	# pragma unroll
	for(int i = tid; i < packed_ncols; i+=block_size) {
		//  each thread add 5 values, 
		unsigned int packed_sum = 0;
		for(int j = 0; j < 5; j++) {
			if (i * 5 + j >= ncols) {
				break;
			} else {
				packed_sum += pow[j] * __bfloat162uint_rn(
					src[row_id * ncols + i * 5 + j]
				);
			}
		}
		
		// Since the max of 5 digit ternary is 242, we don't need to remap 
		// to 0 - 255.
		dst[row_id * packed_ncols + i] = static_cast<uint8_t>(packed_sum);
	}
}
