


#include "cpu_subs.h"
#include "subroutines.cuh"



int main()
{

	// timing variables
	float elapsed_time_ms = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	// assign variables
	int nreps = 50;		// repeated counts, to average execution time


	//**********************************************************//
	// data size
	unsigned data_size = 8 * 1024 * 1024;	// 4 Megabyte data

	// int data_size = 256 - 1;
	//**********************************************************//

	long nbytes = data_size*sizeof(double);
	double gb = nbytes / (double)1e9;

	// host data
	double *h_data = 0;
	h_data = (double*)malloc(nbytes);
	printf("allocated %.2f MB on CPU\n", nbytes / (1024.f*1024.f));

	for (unsigned i = 0; i < data_size; i++)
		h_data[i] = 1.0f + i;


	// device data
	double *d_data = 0;
	cudaMalloc( (void**)&d_data, nbytes );
	printf("allocated %.2f MB on GPU\n", nbytes/(1024.f*1024.f) );


	cudaEventRecord(start, 0);
	cudaMemcpy(d_data, h_data, nbytes, cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	elapsed_time_ms /= nreps;
	printf("host to device transfer bandWidth: %f\n", gb / elapsed_time_ms * 1e3);




	// actual computation
	double result_cpu = 0.0;
	elapsed_time_ms = timing_experiment_cpu(reduction_cpu, h_data, data_size, nreps, &result_cpu);
	printf("CPU reduction  (kernel 00): execution = %8.4f,  bandwidth = %8.4f Gb/s,  result = %f\n", elapsed_time_ms / nreps, gb * nreps / elapsed_time_ms * 1e3, result_cpu);



	double result_CUDA = 0.0;


	elapsed_time_ms = 0.0f;
	cudaEventRecord(start, 0);
	for (int i = 0; i < nreps; i++){
		result_CUDA = sum_gpu(d_data, data_size);
	}
	cudaEventRecord(stop, 0);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	printf("CUDA reduction (from book): execution = %8.4f,  bandwidth = %8.4f Gb/s,  result = %f\n", elapsed_time_ms / nreps, gb * nreps / elapsed_time_ms * 1e3, result_CUDA);


	elapsed_time_ms = 0.0f;
	cudaEventRecord(start, 0);
	for (int i = 0; i < nreps; i++){
		result_CUDA = sum_gpu_reduce_full(kernel_reduce_score10, d_data, data_size, 0);
	}
	cudaEventRecord(stop, 0);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	printf("CUDA reduction (kernel 10): execution = %8.4f,  bandwidth = %8.4f Gb/s,  result = %f\n", elapsed_time_ms / nreps, gb * nreps / elapsed_time_ms * 1e3, result_CUDA);




	elapsed_time_ms = 0.0f;
	cudaEventRecord(start, 0);
	for (int i = 0; i < nreps; i++){
		result_CUDA = sum_gpu_reduce_full(kernel_reduce_score20, d_data, data_size, 0);
	}
	cudaEventRecord(stop, 0);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	printf("CUDA reduction (kernel 20): execution = %8.4f,  bandwidth = %8.4f Gb/s,  result = %f\n", elapsed_time_ms / nreps, gb * nreps / elapsed_time_ms * 1e3, result_CUDA);





	elapsed_time_ms = 0.0f;
	cudaEventRecord(start, 0);
	for (int i = 0; i < nreps; i++){
		result_CUDA = sum_gpu_reduce_full(kernel_reduce_score30, d_data, data_size, 0);
	}
	cudaEventRecord(stop, 0);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	printf("CUDA reduction (kernel 30): execution = %8.4f,  bandwidth = %8.4f Gb/s,  result = %f\n", elapsed_time_ms / nreps, gb * nreps / elapsed_time_ms * 1e3, result_CUDA);





	elapsed_time_ms = 0.0f;
	cudaEventRecord(start, 0);
	for (int i = 0; i < nreps; i++){
		result_CUDA = sum_gpu_reduce_half_wrap(kernel_reduce_half_score40, d_data, data_size, 0);
	}
	cudaEventRecord(stop, 0);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	printf("CUDA reduction (kernel 40): execution = %8.4f,  bandwidth = %8.4f Gb/s,  result = %f\n", elapsed_time_ms / nreps, gb * nreps / elapsed_time_ms * 1e3, result_CUDA);




	elapsed_time_ms = 0.0f;
	cudaEventRecord(start, 0);
	for (int i = 0; i < nreps; i++){
		result_CUDA = sum_gpu_reduce_half_wrap(kernel_reduce_half_score50, d_data, data_size, 0);
	}
	cudaEventRecord(stop, 0);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	printf("CUDA reduction (kernel 50): execution = %8.4f,  bandwidth = %8.4f Gb/s,  result = %f\n", elapsed_time_ms / nreps, gb * nreps / elapsed_time_ms * 1e3, result_CUDA);





	elapsed_time_ms = 0.0f;
	cudaEventRecord(start, 0);
	for (int i = 0; i < nreps; i++){
		result_CUDA = sum_reduce_recursive_cuda(d_data, data_size, 0);
	}
	cudaEventRecord(stop, 0);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	printf("CUDA reduction (kernel 60): execution = %8.4f,  bandwidth = %8.4f Gb/s,  result = %f\n", elapsed_time_ms / nreps, gb * nreps / elapsed_time_ms * 1e3, result_CUDA);





	printf("\nCUDA: %s\n", cudaGetErrorString( cudaGetLastError() ) );


	if (d_data)
		cudaFree(d_data);
	if (h_data)
		free(h_data);


	// cudaThreadExit();

	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	getchar();

	return 0;
}