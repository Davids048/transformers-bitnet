#include <cuda_bf16.h> 
#include <stdio.h> 
#include <math.h> 

typedef unsigned char uint8_t;

extern "C" 
__global__ void unpack_sub2bits (
	const uint8_t *__restrict__ src,
	__nv_bfloat16 *__restrict__ dst,
	int nrows,
	int ncols,
	int packed_ncols
) {
	int row_id = blockIdx.x;
	int tid = threadIdx.x;
	int block_size = blockDim.x;

	// Each block handles one row 
	// Each thread handles 1 read, and (at most) 5 writes

	# pragma unroll 
	for (int i = tid; i < packed_ncols; i += block_size) {
		unsigned int packed_sum = static_cast<unsigned int>(
			src[row_id * packed_ncols + i]
		);

		// Get the five values out of the packed sum. Then write each to 
		// its corresponding location. 
		# pragma unroll 
		for (int j = 0; j < 5; j++) {
			if (i * 5 + j >= ncols) {
				break;
			} else {
				dst[row_id * ncols + i * 5 + j] = __uint2bfloat16_rn(
					packed_sum % 3
				);
				packed_sum /= 3;
			}
		}
	}
}
