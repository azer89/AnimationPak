
#include "CUDAWorker.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <time.h> // time seed
#include <stdlib.h>     /* srand, rand */
#include <time.h> 

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

CUDAWorker::CUDAWorker()
{
	//TestCUDA();
	int nDevices;

	std::cout << "===== CUDA =====\n";
	
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) 
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		//printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		//printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		//printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
		int maxThreadPerBlock = prop.maxThreadsPerBlock;
		int maxGridSize = prop.maxGridSize[0];
		
		std::cout << "  maxThreadsPerBlock: " << maxThreadPerBlock <<"\n";
		std::cout << "  maxGridSize: " << maxGridSize << "\n";
	}
	std::cout << "================\n";
}

CUDAWorker::~CUDAWorker()
{

}

int CUDAWorker::TestCUDA()
{
	/*const int arraySize = 20;
	const int a[arraySize] = { 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,  1, 2, 3, 4, 5, 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50, 10, 20, 30, 40, 50, 10, 20, 30, 40, 50, 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };*/

	/*const int arraySize = 1000;
	int a[arraySize] = { 0 };
	int b[arraySize] = { 0 };
	int c[arraySize] = { 0 };
	*/

	// dynamic
	int arraySize = 1024;
	int* a = new int[arraySize];
	int* b = new int[arraySize];
	int* c = new int[arraySize];
	   
	for (int i = 0; i < arraySize; i++)
	{
		a[i] = rand() % 100;
		b[i] = rand() % 100;
	}

	//std::cout << "test\n";

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	/*for (int i = 0; i < arraySize; i++)
	{
		std::cout << i << " --> " << a[i] << " + " << b[i] << " = " << c[i] << "\n";
	}*/


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> > (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "--> addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}