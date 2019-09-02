
#include "CUDAWorker.cuh"


#include <stdio.h>
#include <iostream>

#include <time.h> // time seed
#include <stdlib.h>     /* srand, rand */
#include <time.h> 


#include "StuffWorker.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


__device__
A3DVectorGPU operator+(const A3DVectorGPU& p, const A3DVectorGPU& v)
{
	return A3DVectorGPU(p._x + v._x, p._y + v._y, p._z + v._z);
}

__device__
A3DVectorGPU operator-(const A3DVectorGPU& p, const A3DVectorGPU& v)
{
	return A3DVectorGPU(p._x - v._x, p._y - v._y, p._z - v._z);
}

__device__
A3DVectorGPU operator*(const A3DVectorGPU& p, const float& f)
{
	return A3DVectorGPU(p._x * f, p._y * f, p._z * f);
}

__device__
A3DVectorGPU operator/(const A3DVectorGPU& p, const float& f)
{
	return A3DVectorGPU(p._x / f, p._y / f, p._z / f);
}


// length of a vector
__device__
float Length(const A3DVectorGPU& p) {
	return sqrt(p._x * p._x +
		p._y * p._y +
		p._z * p._z);
}

__device__
A3DVectorGPU Norm(const A3DVectorGPU& p) // get the unit vector
{
	float vlength = sqrt(p._x * p._x + p._y * p._y + p._z * p._z);

	if (vlength == 0) { return A3DVectorGPU(0, 0, 0); }

	return A3DVectorGPU(p._x / vlength,
		p._y / vlength,
		p._z / vlength);
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void Simulate_GPU(A3DVectorGPU* pos_array, 
	                     A3DVectorGPU* velocity_array, 
	                     A3DVectorGPU* edge_force_array,
						 A3DVectorGPU* z_force_array,
						 A3DVectorGPU* repulsion_force_array,
						 A3DVectorGPU* boundary_force_array,
						 A3DVectorGPU* overlap_force_array,
						 A3DVectorGPU* rotation_force_array, 
						 int n, 
						 float dt, 
	                     float velocity_cap_dt)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
	{
		// oiler
		velocity_array[i] = velocity_array[i] +

			                ((edge_force_array[i] +
							z_force_array[i] +
							repulsion_force_array[i] +
							boundary_force_array[i] +
							overlap_force_array[i] +
							rotation_force_array[i]) * dt);

		float len = Length(velocity_array[i]);

		if (len > velocity_cap_dt)
		{
			velocity_array[i] = Norm(velocity_array[i]) * velocity_cap_dt;
		}

		pos_array[i] = pos_array[i] + velocity_array[i] * dt;
	}
}

void CUDAWorker::Simulate(float dt, float velocity_cap)
{
	int blockSize = SystemParams::_cuda_block_size;
	int numBlocks = (_num_vertex + blockSize - 1) / blockSize;
	Simulate_GPU <<<numBlocks, blockSize >>> (_pos_array,
		_velocity_array,
		_edge_force_array,
		_z_force_array,
		_repulsion_force_array,
		_boundary_force_array,
		_overlap_force_array,
		_rotation_force_array,
		_num_vertex,
		dt,
		velocity_cap * dt);

	cudaDeviceSynchronize();
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

	/*
	_edge_force_array = 0;
	_z_force_array = 0;
	_repulsion_force_array = 0;
	_boundary_force_array = 0;
	_overlap_force_array = 0;
	_rotation_force_array = 0;
	*/
}

CUDAWorker::~CUDAWorker()
{
	cudaFree(_edge_force_array);
	cudaFree(_z_force_array);
	cudaFree(_repulsion_force_array);
	cudaFree(_boundary_force_array);
	cudaFree(_overlap_force_array);
	cudaFree(_rotation_force_array);
}

void CUDAWorker::InitCUDA(int num_vertex)
{
	_num_vertex = num_vertex;

	// mass positions
	cudaMallocManaged(&_pos_array, num_vertex * sizeof(A3DVectorGPU));

	// mass velocities
	cudaMallocManaged(&_velocity_array, num_vertex * sizeof(A3DVectorGPU));
	
	// mass forces
	cudaMallocManaged(&_edge_force_array, num_vertex * sizeof(A3DVectorGPU));
	cudaMallocManaged(&_z_force_array, num_vertex * sizeof(A3DVectorGPU));
	cudaMallocManaged(&_repulsion_force_array, num_vertex * sizeof(A3DVectorGPU));
	cudaMallocManaged(&_boundary_force_array, num_vertex * sizeof(A3DVectorGPU));
	cudaMallocManaged(&_overlap_force_array, num_vertex * sizeof(A3DVectorGPU));
	cudaMallocManaged(&_rotation_force_array, num_vertex * sizeof(A3DVectorGPU));

	/*for (unsigned int a = 0; a < _num_vertex; a++)
	{
		_edge_force_array[a] = A3DVectorGPU();
	}*/
	
}

void CUDAWorker::RetrievePositionAndVelocityData()
{
	int idx = 0;
	for (unsigned int a = 0; a < StuffWorker::_element_list.size(); a++)
	{
		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._massList.size(); b++)
		{
			CopyVector_GPU2CPU(&_pos_array[idx], StuffWorker::_element_list[a]._massList[b]._pos);
			CopyVector_GPU2CPU(&_velocity_array[idx], StuffWorker::_element_list[a]._massList[b]._velocity);
			idx++;
		}
	}
}

void CUDAWorker::SendPositionAndVelocityData()
{
	int idx = 0;
	for (unsigned int a = 0; a < StuffWorker::_element_list.size(); a++)
	{
		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._massList.size(); b++)
		{
			CopyVector_CPU2GPU(StuffWorker::_element_list[a]._massList[b]._pos, &_pos_array[idx]);
			CopyVector_CPU2GPU(StuffWorker::_element_list[a]._massList[b]._velocity, &_velocity_array[idx]);
			idx++;
		}
	}
}

void CUDAWorker::SendForceData()
{
	int idx = 0;
	for (unsigned int a = 0; a < StuffWorker::_element_list.size(); a++)
	{
		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._massList.size(); b++)
		{
			CopyVector_CPU2GPU(StuffWorker::_element_list[a]._massList[b]._edgeForce,      &_edge_force_array[idx]);
			CopyVector_CPU2GPU(StuffWorker::_element_list[a]._massList[b]._zForce,         &_z_force_array[idx]);
			CopyVector_CPU2GPU(StuffWorker::_element_list[a]._massList[b]._repulsionForce, &_repulsion_force_array[idx]);
			CopyVector_CPU2GPU(StuffWorker::_element_list[a]._massList[b]._boundaryForce,  &_boundary_force_array[idx]);
			CopyVector_CPU2GPU(StuffWorker::_element_list[a]._massList[b]._overlapForce,   &_overlap_force_array[idx]);
			CopyVector_CPU2GPU(StuffWorker::_element_list[a]._massList[b]._rotationForce,  &_rotation_force_array[idx]);

			idx++;
		}
	}

	// test, delete me
	/*idx = 0;
	for (unsigned int a = 0; a < _num_vertex; a++)
	{
		std::cout << _edge_force_array[a]._x << ", " << _edge_force_array[a]._y << ", " << _edge_force_array[a]._z << "\n";
	}

	std::cout << "done\n";*/
}



void CUDAWorker::CopyVector_CPU2GPU(const A3DVector& src, A3DVectorGPU* dest)
{
	dest->_x = src._x;
	dest->_y = src._y;
	dest->_z = src._z;
}

void CUDAWorker::CopyVector_GPU2CPU(A3DVectorGPU* src, A3DVector& dest)
{
	dest._x = src->_x;
	dest._y = src->_y;
	dest._z = src->_z;
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
	addKernel <<<1, size >>> (dev_c, dev_a, dev_b);

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