#include "subroutines.cuh"


//-----------------------------------------------------------------------//
//* Some common small functions and constants *//
//-----------------------------------------------------------------------//

const int constSharedMemSize = 256;

const int maxThreads = 256;  // number of threads per block

#define imin(a,b) (a<b?a:b)

bool isPow2(unsigned int x) {
	return ((x&(x - 1)) == 0);
}

unsigned int nextPow2(unsigned int x) {
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

__global__ void add_two_values_in_gpu(double *d_input, double *d_res)
{
	int tid = threadIdx.x;

	if (tid == 0)
		d_res[0] = d_input[0] + d_res[1];
}

__global__ void access_value_in_gpu(double *d_input, double *d_res, int index)
{
	int tid = threadIdx.x;

	if (tid == 0)
		d_res[0] = d_input[index];
}



//-----------------------------------------------------------------------//
//* Reduce from cuda book	*//
//-----------------------------------------------------------------------//

__global__ void kernel_sum(double *d_data, double *d_result, int data_size)
{
	__shared__ double s_data[constSharedMemSize];

	long index = threadIdx.x + blockIdx.x * blockDim.x;
	int id = threadIdx.x;

	s_data[id] = 0.0;
	if (index < data_size)
		s_data[id] = d_data[index];
	__syncthreads();

	int step = constSharedMemSize / 2;
	while (step != 0) {
		if (id < step)
			s_data[id] += s_data[id + step];
		__syncthreads();
		step /= 2;
	}

	if (id == 0)
		d_result[blockIdx.x] = s_data[0];
}


double sum_gpu(double *d_data, int data_size){

	int threadsPerBlock = constSharedMemSize;
	int numBlocks = imin(65536, (data_size + constSharedMemSize - 1) / constSharedMemSize);

	double *h_result;
	double *d_result;

	h_result = (double*)malloc(numBlocks*sizeof(double));
	cudaMalloc((void**)&d_result, numBlocks*sizeof(double));

	kernel_sum << <numBlocks, threadsPerBlock >> >(d_data, d_result, data_size);
	cudaMemcpy(h_result, d_result, numBlocks*sizeof(double), cudaMemcpyDeviceToHost);

	double sum = 0.0;
	for (int j = 0; j < numBlocks; j++){
		sum += h_result[j];
	}

	free(h_result);
	cudaFree(d_result);

	return sum;
}




//----------------------------------------------------------------------------//
//* Three kernels of reduce algorithms using threads matching the data size  *//
//----------------------------------------------------------------------------//

__global__ void kernel_reduce_score10(double *d_input, double *d_res, int size)
{
	__shared__ double s_data[constSharedMemSize];

	long index = threadIdx.x + blockIdx.x*blockDim.x;
	int tid = threadIdx.x;

	s_data[tid] = 0.0;
	if (index < size)
		s_data[tid] = d_input[index];
	__syncthreads();

	for (int s = 2; s <= blockDim.x; s = s * 2)
	{
		if ((tid % s) == 0)
			s_data[tid] += s_data[tid + s / 2];
		__syncthreads();
	}

	if (tid == 0){
		d_res[blockIdx.x] = s_data[0];
	}
}


// replace % operator, avoid highly divergent wraps and slow operators
__global__ void kernel_reduce_score20(double *d_input, double *d_res, int size)
{
	__shared__ double s_data[constSharedMemSize];

	long index = threadIdx.x + blockIdx.x*blockDim.x;
	int tid = threadIdx.x;

	s_data[tid] = 0.0;
	if (index < size)
		s_data[tid] = d_input[index];
	__syncthreads();

	for (int s = 2; s <= blockDim.x; s = s * 2){
		index = tid * s;
		if (index < blockDim.x)
			s_data[index] += s_data[index + s / 2];
		__syncthreads();
	}

	if (tid == 0)
		d_res[blockIdx.x] = s_data[0];
}


// deal with shared memory bank conflicts
__global__ void kernel_reduce_score30(double *d_input, double *d_res, int size)
{
	__shared__ double s_data[constSharedMemSize];

	long index = threadIdx.x + blockIdx.x*blockDim.x;
	int tid = threadIdx.x;

	s_data[tid] = 0.0;
	if (index < size)
		s_data[tid] = d_input[index];
	__syncthreads();

	for (int s = blockDim.x / 2; s >= 1; s = s >> 1){
		if (tid < s)
			s_data[tid] += s_data[tid + s];
		__syncthreads();
	}

	if (tid == 0)
		d_res[blockIdx.x] = s_data[tid];
}


double sum_gpu_reduce_full(void(*kernel)(double*, double*, int), double *d_input, int size, int flag)
{
	//int threadsPerBlock = constSharedMemSize;
	//int numBlocks = imin(65536, (size + constSharedMemSize - 1) / constSharedMemSize);

	int threadsPerBlock = constSharedMemSize;
	int numBlocks = imin(65536, (nextPow2(size - 1) + constSharedMemSize - 1) / constSharedMemSize);

	double *d_result;
	cudaMalloc((void**)&d_result, numBlocks * sizeof(double));

	kernel << <numBlocks, threadsPerBlock >> >(d_input, d_result, size);
	cudaDeviceSynchronize();

	if (numBlocks == 1){
		double h_result;
		cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

		if (flag)
			cudaFree(d_result);

		return h_result;
	}
	return sum_gpu_reduce_full(kernel, d_result, numBlocks, 1);
}




//----------------------------------------------------------------------------//
//* Three kernels of reduce algorithms using half threads of data size  *//
//----------------------------------------------------------------------------//

__device__ void warpReduce(volatile double* s_data, int tid) {
	s_data[tid] += s_data[tid + 32];
	s_data[tid] += s_data[tid + 16];
	s_data[tid] += s_data[tid + 8];
	s_data[tid] += s_data[tid + 4];
	s_data[tid] += s_data[tid + 2];
	s_data[tid] += s_data[tid + 1];
}

// deal with the first loop of idea threads
__global__ void kernel_reduce_half_score40(double *d_input, double *d_res, int size)
{
	__shared__ double s_data[constSharedMemSize];

	long index = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;

	/*double mySum = (index < size / 2) ? d_input[index] : 0.0f;
	if (index + size / 2 < size)
		mySum += d_input[index + size / 2];
	s_data[tid] = mySum;
	__syncthreads();*/

	s_data[tid] = 0.0;
	if (index < size / 2)
		s_data[tid] = d_input[index] + d_input[index + size / 2];
	__syncthreads();

	for (int s = blockDim.x / 2; s >= 1; s = s >> 1){
		if (tid < s)
			s_data[tid] += s_data[tid + s];
		__syncthreads();
	}

	if (tid == 0)
		d_res[blockIdx.x] = s_data[tid];
}


// deal with first loop and uproll last wrap
__global__ void kernel_reduce_half_score50(double *d_input, double *d_res, int size)
{
	__shared__ volatile double s_data[constSharedMemSize];

	long index = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;

	/*
	double mySum = (index < size / 2) ? d_input[index] : 0.0f;
	if (index < size / 2)
		mySum += d_input[index + size / 2];
	s_data[tid] = mySum;
	__syncthreads();*/

	s_data[tid] = 0.0;
	if (index < size / 2)
		s_data[tid] = d_input[index] + d_input[index + size / 2];
	__syncthreads();

	for (int s = blockDim.x / 2; s >= 64; s = s >> 1){
		if (tid < s)
			s_data[tid] += s_data[tid + s];
		__syncthreads();
	}

	if (tid < 32)
		warpReduce(s_data, tid);

	if (tid == 0)
		d_res[blockIdx.x] = s_data[tid];
}



double sum_gpu_reduce_half(void(*kernel)(double*, double*, int), double *d_input, int size, int flag)
{
	//int threadsPerBlock = constSharedMemSize;
	//int numBlocks = imin(65536, (size / 2 + constSharedMemSize - 1) / constSharedMemSize);

	int threadsPerBlock = constSharedMemSize;
	int numBlocks = imin(65536, (nextPow2((size - 1) / 2) + constSharedMemSize - 1) / constSharedMemSize);

	double *d_result;
	cudaMalloc((void**)&d_result, numBlocks * sizeof(double));

	kernel << <numBlocks, threadsPerBlock >> >(d_input, d_result, size);
	cudaDeviceSynchronize();

	if (numBlocks == 1){
		double h_result;
		cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

		if (flag)
			cudaFree(d_result);

		return h_result;
	}
	return sum_gpu_reduce_half(kernel, d_result, numBlocks, 1);
}


double sum_gpu_reduce_half_wrap(void(*kernel)(double*, double*, int), double *d_input, int size, int flag){
	
	double value = sum_gpu_reduce_half(kernel, d_input, size, flag);
	
	double value_gpu = 0.0;
	if (size & 1) // checked if size is odd number
	{
		double *d_value;
		cudaMalloc((void**)&d_value, sizeof(double));
		access_value_in_gpu << <1, 1 >> >(d_input, d_value, size - 1);

		cudaMemcpy(&value_gpu, d_value, sizeof(double), cudaMemcpyDeviceToHost);
	}

	return value + value_gpu;
}



//----------------------------------------------------------------------------//
//* Complete unroll last warp, using template  *//
//----------------------------------------------------------------------------//

template <unsigned int blockSize>
__device__ void warpReduce2(volatile double *sdata, int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}


template <unsigned int blockSize>  //, bool nIsPow2
__global__ void reduce_kernel(double *d_input, double *d_res, int size) {

	extern __shared__ volatile double sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * (blockSize * 2) + tid;

	int gridSize = blockSize * 2 * gridDim.x;

	double mySum = 0.0;
	while (i < size) {
		mySum += d_input[i];
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (i + blockSize < size)
			mySum += d_input[i + blockSize];
		i += gridSize;
	}
	sdata[tid] = mySum;
	__syncthreads();

	/*sdata[tid] = 0;
	while (i < size) {
		sdata[tid] += d_input[i] + d_input[i + blockSize];
		i += gridSize; 
	}
	__syncthreads();*/

	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

	if (tid < 32) warpReduce2<blockSize>(sdata, tid);

	if (tid == 0) d_res[blockIdx.x] = sdata[0];
}


double sum_reduce_recursive_cuda(double *d_data, int size, int flag){

	int threads = (size < maxThreads * 2) ? nextPow2((size + 1) / 2) : maxThreads;
	int blocks = (size + (threads * 2 - 1)) / (threads * 2);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds

	int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

	double *d_result;
	cudaMalloc((void**)&d_result, blocks*sizeof(double));

	switch (threads)
	{
	case 512:
		reduce_kernel<512> << < blocks, threads, smemSize >> >(d_data, d_result, size); break;
	case 256:
		reduce_kernel<256> << < blocks, threads, smemSize >> >(d_data, d_result, size); break;
	case 128:
		reduce_kernel<128> << < blocks, threads, smemSize >> >(d_data, d_result, size); break;
	case 64:
		reduce_kernel< 64> << < blocks, threads, smemSize >> >(d_data, d_result, size); break;
	case 32:
		reduce_kernel< 32> << < blocks, threads, smemSize >> >(d_data, d_result, size); break;
	case 16:
		reduce_kernel< 16> << < blocks, threads, smemSize >> >(d_data, d_result, size); break;
	case 8:
		reduce_kernel< 8> << < blocks, threads, smemSize >> >(d_data, d_result, size); break;
	case 4:
		reduce_kernel< 4> << < blocks, threads, smemSize >> >(d_data, d_result, size); break;
	case 2:
		reduce_kernel< 2> << < blocks, threads, smemSize >> >(d_data, d_result, size); break;
	case 1:
		reduce_kernel< 1> << < blocks, threads, smemSize >> >(d_data, d_result, size); break;
	}
	cudaDeviceSynchronize();

	if (blocks == 1){
		double h_result;
		cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
		if (flag){
			cudaFree(d_result);
		}
		return h_result;
	}

	return sum_reduce_recursive_cuda(d_result, blocks, 1);
}

