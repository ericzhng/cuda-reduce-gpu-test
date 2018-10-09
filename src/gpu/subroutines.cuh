
#include <stdio.h>

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>


// kernel of summation from book

__global__ void kernel_sum(double *d_data, double *d_result, int data_size);

double sum_gpu(double *d_data, int data_size);




// Original kernel_A, used to generate the reference result

__global__ void kernel_reduce_score10(double *d_input, double *d_res, int size);
__global__ void kernel_reduce_score20(double *d_input, double *d_res, int size);
__global__ void kernel_reduce_score30(double *d_input, double *d_res, int size);

double sum_gpu_reduce_full(void(*kernel)(double*, double*, int), double *d_data, int data_size, int flag);


__global__ void kernel_reduce_half_score40(double *d_input, double *d_res, int size);
__global__ void kernel_reduce_half_score50(double *d_input, double *d_res, int size);

double sum_gpu_reduce_half(void(*kernel)(double*, double*, int), double *d_data, int data_size, int flag);

double sum_gpu_reduce_half_wrap(void(*kernel)(double*, double*, int), double *d_input, int size, int flag);


double sum_reduce_recursive_cuda(double *d_data, int size, int flag);

bool isPow2(unsigned int x);

unsigned int nextPow2(unsigned int x);