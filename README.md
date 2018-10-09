# Summation by reduction in GPU using CUDA

Reduction is popularly used for summation in either cpu or gpu code. Comparisons between different implementations of reduction code are done using CUDA C/C++.

### Note

The ideas are referred from [1] where the author discussed in detail the implementations. This code is to verify the implementation for data size of power of 2 or not.


### Issues

The test is done using data size of power of 2. If the data size is not power of 2, then some issues might arise. It could be that the improved reduce methods won't work well as reference version of code. The improved version has to use a larger size with a power of 2.

We can adjust the performance in following two ways:
[1] Divide the data size into a size of power of 2, and one that is not. Apply reduce to each set independently. 
[2] Change the threads and blocks assignment, most of the time, it works, sometimes, it doesn't.
[3] Leave the other part of data in cpu reduction, if that is a small set.

### Reference

[1](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
