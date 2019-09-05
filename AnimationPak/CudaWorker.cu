
#include "CUDAWorker.cuh"


#include <stdio.h>
#include <iostream>

#include <time.h> // time seed
#include <stdlib.h>     /* srand, rand */
#include <time.h> 


#include "StuffWorker.h"

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
	return sqrtf(p._x * p._x +
		p._y * p._y +
		p._z * p._z);
}

__device__
A3DVectorGPU Norm(const A3DVectorGPU& p) // get the unit vector
{
	float vlength = sqrtf(p._x * p._x + p._y * p._y + p._z * p._z);

	if (vlength == 0) { return A3DVectorGPU(0, 0, 0); }

	return A3DVectorGPU(p._x / vlength,
		p._y / vlength,
		p._z / vlength);
}

__device__ A3DVectorGPU DirectionTo(const A3DVectorGPU& a, const A3DVectorGPU& b)
{
	return A3DVectorGPU(b._x - a._x,
			            b._y - a._y,
			            b._z - a._z);
}

__device__ 	void GetUnitAndDist(const A3DVectorGPU& p, A3DVectorGPU& unitVec, float& dist)
{
	dist = sqrtf(p._x * p._x + p._y * p._y + p._z * p._z);

	unitVec = A3DVectorGPU(p._x / dist,
							p._y / dist,
							p._z / dist);
}

__global__ void SolveForSprings3D_GPU(SpringGPU* spring_array,
									A3DVectorGPU* pos_array,
									A3DVectorGPU* edge_force_array,
	                                float* spring_parameters,
									int n_springs)
{
	A3DVectorGPU pt0;
	A3DVectorGPU pt1;
	A3DVectorGPU dir;
	A3DVectorGPU dir_not_unit;
	A3DVectorGPU eForce;
	float dist = 0;
	float diff = 0;
	float k = 0;
	int idx0, idx1;
	int spring_type;

	// TODO: Nasty code here
	//float scale_threshold = 1.0f;
	//float magic_number = 3.0f;

	// for squared forces
	float signVal = 1;

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n_springs; i += stride)
	{
		spring_type = spring_array[i]._spring_type;
		k = spring_parameters[spring_type];

		// check this!
		idx0 = spring_array[i]._index0;
		idx1 = spring_array[i]._index1;

		//dir_not_unit = _massList[idx0]._pos.DirectionTo(_massList[idx1]._pos);
		//dir_not_unit.GetUnitAndDist(dir, dist);
		dir_not_unit = DirectionTo(pos_array[idx0], pos_array[idx1]);
		GetUnitAndDist(dir_not_unit, dir, dist);

		diff = dist - spring_array[i]._dist;

		// squared version
		signVal = 1;
		if (diff < 0) { signVal = -1; }

		eForce = dir * k *  diff * diff * signVal;

		edge_force_array[idx0] = edge_force_array[idx0] + eForce;
		edge_force_array[idx1] = edge_force_array[idx1] - eForce;
	}
}

__global__ void ResetForces_GPU(A3DVectorGPU* edge_force_array,
								A3DVectorGPU* z_force_array,
								A3DVectorGPU* repulsion_force_array,
								A3DVectorGPU* boundary_force_array,
								A3DVectorGPU* overlap_force_array,
								A3DVectorGPU* rotation_force_array,
								int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
	{
		edge_force_array[i] = A3DVectorGPU(0, 0, 0);
		z_force_array[i] = A3DVectorGPU(0, 0, 0);
		repulsion_force_array[i] = A3DVectorGPU(0, 0, 0);
		boundary_force_array[i] = A3DVectorGPU(0, 0, 0);
		overlap_force_array[i] = A3DVectorGPU(0, 0, 0);
		rotation_force_array[i] = A3DVectorGPU(0, 0, 0);
	}
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

void CUDAWorker::SolveForSprings3D()
{
	for (unsigned int a = 0; a < _num_vertex; a++)
	{
		_edge_force_array[a] = A3DVectorGPU(0, 0, 0);
	}

	int blockSize = SystemParams::_cuda_block_size;
	int numBlocks = (_num_vertex + blockSize - 1) / blockSize;
	SolveForSprings3D_GPU<<<numBlocks, blockSize >>>(_spring_array, // need data from CPU
		                                             _pos_array,    // need data from CPU
													 _edge_force_array,
													 _spring_parameters,
													 _num_spring);

	cudaDeviceSynchronize(); // must sync
}

void CUDAWorker::ResetForces()
{
	int blockSize = SystemParams::_cuda_block_size;
	int numBlocks = (_num_vertex + blockSize - 1) / blockSize;

	ResetForces_GPU<<<numBlocks, blockSize >>>(_edge_force_array,
		_z_force_array,
		_repulsion_force_array,
		_boundary_force_array,
		_overlap_force_array,
		_rotation_force_array,
		_num_vertex);

	cudaDeviceSynchronize(); // must sync
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

	cudaDeviceSynchronize();// must sync
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

	cudaFree(_pos_array);
	cudaFree(_velocity_array);
	cudaFree(_spring_array);
	cudaFree(_spring_parameters);
}

void CUDAWorker::InitCUDA(int num_vertex, int num_spring)
{
	_num_vertex = num_vertex;
	_num_spring = num_spring;

	// mass positions
	cudaMallocManaged(&_pos_array, _num_vertex * sizeof(A3DVectorGPU));

	// mass velocities
	cudaMallocManaged(&_velocity_array, _num_vertex * sizeof(A3DVectorGPU));
	
	// mass forces
	cudaMallocManaged(&_edge_force_array,      _num_vertex * sizeof(A3DVectorGPU));
	cudaMallocManaged(&_z_force_array,         _num_vertex * sizeof(A3DVectorGPU));
	cudaMallocManaged(&_repulsion_force_array, _num_vertex * sizeof(A3DVectorGPU));
	cudaMallocManaged(&_boundary_force_array,  _num_vertex * sizeof(A3DVectorGPU));
	cudaMallocManaged(&_overlap_force_array,   _num_vertex * sizeof(A3DVectorGPU));
	cudaMallocManaged(&_rotation_force_array,  _num_vertex * sizeof(A3DVectorGPU));

	// springs
	cudaMallocManaged(&_spring_array, _num_spring * sizeof(SpringGPU));

	// spring parameters
	cudaMallocManaged(&_spring_parameters, 4 * sizeof(float));
	_spring_parameters[0] = SystemParams::_k_edge;
	_spring_parameters[1] = SystemParams::_k_time_edge;
	_spring_parameters[2] = SystemParams::_k_edge;
	_spring_parameters[3] = SystemParams::_k_neg_space_edge;

	// call this only once, assuming springs are not changed during simulation
	InitSpringData();
	
}

void CUDAWorker::RetrieveEdgeForceData()
{
	int idx = 0;
	for (unsigned int a = 0; a < StuffWorker::_element_list.size(); a++)
	{
		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._massList.size(); b++)
		{
			CopyVector_GPU2CPU(&_edge_force_array[idx], StuffWorker::_element_list[a]._massList[b]._edgeForce);
			idx++;
		}
	}
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

void CUDAWorker::SendSpringLengths()
{
	int idx = 0;
	for (unsigned int a = 0; a < StuffWorker::_element_list.size(); a++)
	{
		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._layer_springs.size(); b++)
		{
			_spring_array[idx]._dist = StuffWorker::_element_list[a]._layer_springs[b]._dist; // 0
			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._time_springs.size(); b++)
		{
			_spring_array[idx]._dist = StuffWorker::_element_list[a]._time_springs[b]._dist; // 1
			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._auxiliary_springs.size(); b++)
		{
			_spring_array[idx]._dist = StuffWorker::_element_list[a]._auxiliary_springs[b]._dist; // 2
			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._neg_space_springs.size(); b++)
		{
			_spring_array[idx]._dist = StuffWorker::_element_list[a]._neg_space_springs[b]._dist; // 3
			idx++;
		}
	}

	//std::cout << idx << "\n";
}

// call this only once
void CUDAWorker::InitSpringData()
{
	int idx = 0;
	for (unsigned int a = 0; a < StuffWorker::_element_list.size(); a++)
	{
		int mass_offset = StuffWorker::_element_list[a]._cuda_mass_array_offset;

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._layer_springs.size(); b++)
		{
			AnIndexedLine ln = StuffWorker::_element_list[a]._layer_springs[b];
			_spring_array[idx] = SpringGPU(ln._index0 + mass_offset, ln._index1 + mass_offset, ln._dist, 0); // 0
			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._time_springs.size(); b++)
		{
			AnIndexedLine ln = StuffWorker::_element_list[a]._time_springs[b];
			_spring_array[idx] = SpringGPU(ln._index0 + mass_offset, ln._index1 + mass_offset, ln._dist, 1); // 1
			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._auxiliary_springs.size(); b++)
		{
			AnIndexedLine ln = StuffWorker::_element_list[a]._auxiliary_springs[b];
			_spring_array[idx] = SpringGPU(ln._index0 + mass_offset, ln._index1 + mass_offset, ln._dist, 2); // 2
			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._neg_space_springs.size(); b++)
		{
			AnIndexedLine ln = StuffWorker::_element_list[a]._neg_space_springs[b];
			_spring_array[idx] = SpringGPU(ln._index0 + mass_offset, ln._index1 + mass_offset, ln._dist, 3); // 3
			idx++;
		}
	}

	//std::cout << idx << "\n";
}

void CUDAWorker::SendPositionData()
{
	int idx = 0;
	for (unsigned int a = 0; a < StuffWorker::_element_list.size(); a++)
	{
		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._massList.size(); b++)
		{
			CopyVector_CPU2GPU(StuffWorker::_element_list[a]._massList[b]._pos, &_pos_array[idx]);
			//CopyVector_CPU2GPU(StuffWorker::_element_list[a]._massList[b]._velocity, &_velocity_array[idx]);
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
			//CopyVector_CPU2GPU(StuffWorker::_element_list[a]._massList[b]._edgeForce,      &_edge_force_array[idx]);
			
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