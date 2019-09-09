
#include "CUDAWorker.cuh"


#include <stdio.h>
#include <iostream>

#include <time.h> // time seed
#include <stdlib.h>     /* srand, rand */
#include <time.h> 


#include "StuffWorker.h"
#include "CUDAFunctions.cuh"



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
			//CopyVector_GPU2CPU(&_edge_force_array[idx], StuffWorker::_element_list[a]._massList[b]._edgeForce_cuda);
			//idx++;

			StuffWorker::_element_list[a]._massList[b]._edgeForce_cuda._x = _edge_force_array[idx]._x;
			StuffWorker::_element_list[a]._massList[b]._edgeForce_cuda._y = _edge_force_array[idx]._y;
			StuffWorker::_element_list[a]._massList[b]._edgeForce_cuda._z = _edge_force_array[idx]._z;
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
			_spring_array[idx++]._dist = StuffWorker::_element_list[a]._layer_springs[b]._dist; // 0
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._time_springs.size(); b++)
		{
			_spring_array[idx++]._dist = StuffWorker::_element_list[a]._time_springs[b]._dist; // 1
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._auxiliary_springs.size(); b++)
		{
			_spring_array[idx++]._dist = StuffWorker::_element_list[a]._auxiliary_springs[b]._dist; // 2
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._neg_space_springs.size(); b++)
		{
			_spring_array[idx++]._dist = StuffWorker::_element_list[a]._neg_space_springs[b]._dist; // 3
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
			_spring_array[idx++] = SpringGPU(ln._index0 + mass_offset, ln._index1 + mass_offset, ln._dist, 0); // 0
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._time_springs.size(); b++)
		{
			AnIndexedLine ln = StuffWorker::_element_list[a]._time_springs[b];
			_spring_array[idx++] = SpringGPU(ln._index0 + mass_offset, ln._index1 + mass_offset, ln._dist, 1); // 1
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._auxiliary_springs.size(); b++)
		{
			AnIndexedLine ln = StuffWorker::_element_list[a]._auxiliary_springs[b];
			_spring_array[idx++] = SpringGPU(ln._index0 + mass_offset, ln._index1 + mass_offset, ln._dist, 2); // 2
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._neg_space_springs.size(); b++)
		{
			AnIndexedLine ln = StuffWorker::_element_list[a]._neg_space_springs[b];
			_spring_array[idx++] = SpringGPU(ln._index0 + mass_offset, ln._index1 + mass_offset, ln._dist, 3); // 3
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