
#include "CUDAWorker.cuh"


#include <stdio.h>
#include <iostream>

#include <time.h> // time seed
#include <stdlib.h>     /* srand, rand */
#include <time.h> 


#include "StuffWorker.h"
#include "CUDAFunctions.cuh"





void CUDAWorker::ResetForces()
{
	int blockSize = SystemParams::_cuda_block_size;
	int numBlocks = (_num_vertex + blockSize - 1) / blockSize;

	/*ResetForces_GPU<<<numBlocks, blockSize >>>(_edge_force_array,
		_z_force_array,
		_repulsion_force_array,
		_boundary_force_array,
		_overlap_force_array,
		_rotation_force_array,
		_num_vertex);
	*/
	cudaDeviceSynchronize(); // must sync
}

void CUDAWorker::Simulate(float dt, float velocity_cap)
{
	/*int blockSize = SystemParams::_cuda_block_size;
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

	cudaDeviceSynchronize();// must sync*/
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
	cudaFree(_edge_force_array_springs);
	cudaFree(_repulsion_f_combinations);
	//cudaFree(_z_force_array);
	//cudaFree(_repulsion_force_array);
	//cudaFree(_boundary_force_array);
	//cudaFree(_overlap_force_array);
	//cudaFree(_rotation_force_array);

	cudaFree(_pos_array);
	//cudaFree(_velocity_array);
	cudaFree(_spring_array);
	cudaFree(_spring_parameters);
}

void CUDAWorker::InitCUDA(int num_vertex, int num_spring, int num_surface_tri)
{
	_num_vertex = num_vertex;
	_num_spring = num_spring;
	_num_surface_tri = num_surface_tri;

	// mass positions
	cudaMallocManaged(&_pos_array, _num_vertex * sizeof(A3DVector));

	// mass velocities
	//cudaMallocManaged(&_velocity_array, _num_vertex * sizeof(A3DVectorGPU));
	
	// mass forces
	//cudaMallocManaged(&_edge_force_array,      _num_vertex * sizeof(A3DVectorGPU));
	cudaMallocManaged(&_edge_force_array_springs,    _num_spring * sizeof(A3DVector));
	//_repulsion_f_combinations
	cudaMallocManaged(&_repulsion_f_combinations, _num_vertex * _num_surface_tri * sizeof(A3DVector));
	//cudaMallocManaged(&_z_force_array,         _num_vertex * sizeof(A3DVectorGPU));
	//cudaMallocManaged(&_repulsion_force_array, _num_vertex * sizeof(A3DVectorGPU));
	//cudaMallocManaged(&_boundary_force_array,  _num_vertex * sizeof(A3DVectorGPU));
	//cudaMallocManaged(&_overlap_force_array,   _num_vertex * sizeof(A3DVectorGPU));
	//cudaMallocManaged(&_rotation_force_array,  _num_vertex * sizeof(A3DVectorGPU));

	// springs
	cudaMallocManaged(&_spring_array, _num_spring * sizeof(SpringGPU));

	// spring parameters
	cudaMallocManaged(&_spring_parameters, 4 * sizeof(float));
	_spring_parameters[0] = SystemParams::_k_edge;
	_spring_parameters[1] = SystemParams::_k_time_edge;
	_spring_parameters[2] = SystemParams::_k_edge;
	_spring_parameters[3] = SystemParams::_k_neg_space_edge;

	// debug delete me
	//cudaMallocManaged(&_spring_diff_array, _num_spring * sizeof(float));
	//cudaMallocManaged(&_spring_signval_array, _num_spring * sizeof(float));
	//cudaMallocManaged(&_spring_k_array, _num_spring * sizeof(float));
	//cudaMallocManaged(&_spring_mag_array, _num_spring * sizeof(float));
	//cudaMallocManaged(&_spring_dir_array, _num_spring * sizeof(A3DVectorGPU));

	// call this only once, assuming springs are not changed during simulation
	InitSpringData();
	
}

void CUDAWorker::SolveForSprings3D()
{
	// prefetch?
	int device = -1;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(_spring_array, _num_spring * sizeof(SpringGPU), device, NULL);
	cudaMemPrefetchAsync(_edge_force_array_springs, _num_spring * sizeof(A3DVector), device, NULL);


	int blockSize = SystemParams::_cuda_block_size;
	int numBlocks = (_num_spring + blockSize - 1) / blockSize;
	/*
	SolveForSprings3D_GPU(SpringGPU* spring_array, // INPUT
						A3DVectorGPU* pos_array, // INPUT
						A3DVectorGPU* edge_force_array_springs, // OUTPUT
						float* spring_parameters, // INPUT
						int n_springs) // INPUT
	*/
	SolveForSprings3D_GPU <<< numBlocks, blockSize >>> (_spring_array,
		_edge_force_array_springs,
		_spring_parameters,
		_num_spring);

	cudaDeviceSynchronize(); // must sync
}

void CUDAWorker::RetrieveEdgeForceData()
{
	// this f king works
	int idx = 0;
	for (unsigned int a = 0; a < StuffWorker::_element_list.size(); a++)
	//for (unsigned int a = 0; a < _num_spring; a++)
	{
		/*int idx0 = _spring_array[a]._index0;
		int idx1 = _spring_array[a]._index1;

		A3DVector eForce(_edge_force_array_springs[a]._x, _edge_force_array_springs[a]._y, _edge_force_array_springs[a]._z);

		StuffWorker::_element_list[a]._massList[idx0]._edgeForce += eForce;
		StuffWorker::_element_list[a]._massList[idx1]._edgeForce -= eForce;*/

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._layer_springs.size(); b++)
		{
			int idx0 = StuffWorker::_element_list[a]._layer_springs[b]._index0;
			int idx1 = StuffWorker::_element_list[a]._layer_springs[b]._index1;

			A3DVector eForce(_edge_force_array_springs[idx]._x, _edge_force_array_springs[idx]._y, _edge_force_array_springs[idx]._z);
			StuffWorker::_element_list[a]._massList[idx0]._edgeForce += eForce;
			StuffWorker::_element_list[a]._massList[idx1]._edgeForce -= eForce;

			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._time_springs.size(); b++)
		{
			int idx0 = StuffWorker::_element_list[a]._time_springs[b]._index0;
			int idx1 = StuffWorker::_element_list[a]._time_springs[b]._index1;

			A3DVector eForce(_edge_force_array_springs[idx]._x, _edge_force_array_springs[idx]._y, _edge_force_array_springs[idx]._z);
			StuffWorker::_element_list[a]._massList[idx0]._edgeForce += eForce;
			StuffWorker::_element_list[a]._massList[idx1]._edgeForce -= eForce;

			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._auxiliary_springs.size(); b++)
		{
			int idx0 = StuffWorker::_element_list[a]._auxiliary_springs[b]._index0;
			int idx1 = StuffWorker::_element_list[a]._auxiliary_springs[b]._index1;

			A3DVector eForce(_edge_force_array_springs[idx]._x, _edge_force_array_springs[idx]._y, _edge_force_array_springs[idx]._z);
			StuffWorker::_element_list[a]._massList[idx0]._edgeForce += eForce;
			StuffWorker::_element_list[a]._massList[idx1]._edgeForce -= eForce;

			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._neg_space_springs.size(); b++)
		{
			int idx0 = StuffWorker::_element_list[a]._neg_space_springs[b]._index0;
			int idx1 = StuffWorker::_element_list[a]._neg_space_springs[b]._index1;

			A3DVector eForce(_edge_force_array_springs[idx]._x, _edge_force_array_springs[idx]._y, _edge_force_array_springs[idx]._z);
			StuffWorker::_element_list[a]._massList[idx0]._edgeForce += eForce;
			StuffWorker::_element_list[a]._massList[idx1]._edgeForce -= eForce;

			idx++;
		}
	}
}



void CUDAWorker::RetrievePositionAndVelocityData()
{
	/*int idx = 0;
	for (unsigned int a = 0; a < StuffWorker::_element_list.size(); a++)
	{
		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._massList.size(); b++)
		{
			CopyVector_GPU2CPU(&_pos_array[idx], StuffWorker::_element_list[a]._massList[b]._pos);
			CopyVector_GPU2CPU(&_velocity_array[idx], StuffWorker::_element_list[a]._massList[b]._velocity);
			idx++;
		}
	}*/
}

// debug delete me
void CUDAWorker::UnitTestSpringGPU()
{
	/*int idx = 0;
	for (unsigned int a = 0; a < StuffWorker::_element_list.size(); a++)
	{
		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._layer_springs.size(); b++)
		{
			StuffWorker::_element_list[a]._layer_springs[b]._diff_cuda = _spring_diff_array[idx]; // 0
			StuffWorker::_element_list[a]._layer_springs[b]._k_debug_cuda = _spring_k_array[idx]; // 0
			StuffWorker::_element_list[a]._layer_springs[b]._signval_debug_cuda = _spring_signval_array[idx]; // 0
			StuffWorker::_element_list[a]._layer_springs[b]._mag_cuda = _spring_mag_array[idx]; // 0

			StuffWorker::_element_list[a]._layer_springs[b]._dir_cuda._x = _spring_dir_array[idx]._x; // 0
			StuffWorker::_element_list[a]._layer_springs[b]._dir_cuda._y = _spring_dir_array[idx]._y; // 0
			StuffWorker::_element_list[a]._layer_springs[b]._dir_cuda._z = _spring_dir_array[idx]._z; // 0

			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._time_springs.size(); b++)
		{
			StuffWorker::_element_list[a]._time_springs[b]._diff_cuda = _spring_diff_array[idx]; // 1
			StuffWorker::_element_list[a]._time_springs[b]._k_debug_cuda = _spring_k_array[idx]; // 1
			StuffWorker::_element_list[a]._time_springs[b]._signval_debug_cuda = _spring_signval_array[idx]; // 1
			StuffWorker::_element_list[a]._time_springs[b]._mag_cuda = _spring_mag_array[idx]; // 1

			StuffWorker::_element_list[a]._time_springs[b]._dir_cuda._x = _spring_dir_array[idx]._x; // 1
			StuffWorker::_element_list[a]._time_springs[b]._dir_cuda._y = _spring_dir_array[idx]._y; // 1
			StuffWorker::_element_list[a]._time_springs[b]._dir_cuda._z = _spring_dir_array[idx]._z; // 1

			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._auxiliary_springs.size(); b++)
		{
			StuffWorker::_element_list[a]._auxiliary_springs[b]._diff_cuda = _spring_diff_array[idx]; // 2
			StuffWorker::_element_list[a]._auxiliary_springs[b]._k_debug_cuda = _spring_k_array[idx]; // 2
			StuffWorker::_element_list[a]._auxiliary_springs[b]._signval_debug_cuda = _spring_signval_array[idx]; // 2
			StuffWorker::_element_list[a]._auxiliary_springs[b]._mag_cuda = _spring_mag_array[idx]; // 2

			StuffWorker::_element_list[a]._auxiliary_springs[b]._dir_cuda._x = _spring_dir_array[idx]._x; // 2
			StuffWorker::_element_list[a]._auxiliary_springs[b]._dir_cuda._y = _spring_dir_array[idx]._y; // 2
			StuffWorker::_element_list[a]._auxiliary_springs[b]._dir_cuda._z = _spring_dir_array[idx]._z; // 2

			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._neg_space_springs.size(); b++)
		{
			StuffWorker::_element_list[a]._neg_space_springs[b]._diff_cuda          = _spring_diff_array[idx]; // 3
			StuffWorker::_element_list[a]._neg_space_springs[b]._k_debug_cuda       = _spring_k_array[idx]; // 3
			StuffWorker::_element_list[a]._neg_space_springs[b]._signval_debug_cuda = _spring_signval_array[idx]; // 3
			StuffWorker::_element_list[a]._neg_space_springs[b]._mag_cuda = _spring_mag_array[idx]; // 2

			StuffWorker::_element_list[a]._neg_space_springs[b]._dir_cuda._x = _spring_dir_array[idx]._x; // 2
			StuffWorker::_element_list[a]._neg_space_springs[b]._dir_cuda._y = _spring_dir_array[idx]._y; // 2
			StuffWorker::_element_list[a]._neg_space_springs[b]._dir_cuda._z = _spring_dir_array[idx]._z; // 2

			idx++;
		}
	}

	idx = 0;
	for (unsigned int a = 0; a < StuffWorker::_element_list.size(); a++)
	{
		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._layer_springs.size(); b++)
		{
			StuffWorker::_element_list[a]._layer_springs[b].DebugShit();
			//StuffWorker::_element_list[a]._layer_springs[b].PrintKCUDA();
			//StuffWorker::_element_list[a]._layer_springs[b].PrintSignValCUDA();
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._time_springs.size(); b++)
		{
			StuffWorker::_element_list[a]._time_springs[b].DebugShit();
			//StuffWorker::_element_list[a]._time_springs[b].PrintKCUDA();
			//StuffWorker::_element_list[a]._time_springs[b].PrintSignValCUDA();
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._auxiliary_springs.size(); b++)
		{
			StuffWorker::_element_list[a]._auxiliary_springs[b].DebugShit();
			//StuffWorker::_element_list[a]._auxiliary_springs[b].PrintKCUDA();
			//StuffWorker::_element_list[a]._auxiliary_springs[b].PrintSignValCUDA();
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._neg_space_springs.size(); b++)
		{
			StuffWorker::_element_list[a]._neg_space_springs[b].DebugShit();
			//StuffWorker::_element_list[a]._neg_space_springs[b].PrintKCUDA();
			//StuffWorker::_element_list[a]._neg_space_springs[b].PrintSignValCUDA();
		}
	}
	*/
}

void CUDAWorker::SendSpringLengths()
{
	int idx = 0;
	for (unsigned int a = 0; a < StuffWorker::_element_list.size(); a++)
	{
		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._layer_springs.size(); b++)
		{
			_spring_array[idx]._dist = StuffWorker::_element_list[a]._layer_springs[b]._dist; // 0
			_spring_array[idx]._pos0 = _pos_array[_spring_array[idx]._index0];
			_spring_array[idx]._pos1 = _pos_array[_spring_array[idx]._index1];
			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._time_springs.size(); b++)
		{
			_spring_array[idx]._dist = StuffWorker::_element_list[a]._time_springs[b]._dist; // 1
			_spring_array[idx]._pos0 = _pos_array[_spring_array[idx]._index0];
			_spring_array[idx]._pos1 = _pos_array[_spring_array[idx]._index1];
			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._auxiliary_springs.size(); b++)
		{
			_spring_array[idx]._dist = StuffWorker::_element_list[a]._auxiliary_springs[b]._dist; // 2
			_spring_array[idx]._pos0 = _pos_array[_spring_array[idx]._index0];
			_spring_array[idx]._pos1 = _pos_array[_spring_array[idx]._index1];
			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._neg_space_springs.size(); b++)
		{
			_spring_array[idx]._dist = StuffWorker::_element_list[a]._neg_space_springs[b]._dist; // 3
			_spring_array[idx]._pos0 = _pos_array[_spring_array[idx]._index0];
			_spring_array[idx]._pos1 = _pos_array[_spring_array[idx]._index1];
			idx++;
		}
	}

	//std::cout << idx << "\n";

	// unit test (PASSED)
	/*idx = 0;
	for (unsigned int a = 0; a < StuffWorker::_element_list.size(); a++)
	{
		int mass_offset = StuffWorker::_element_list[a]._cuda_mass_array_offset;

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._layer_springs.size(); b++)
		{
			if ((_spring_array[idx]._index0 - StuffWorker::_element_list[a]._layer_springs[b]._index0 - mass_offset) != 0) { std::cout << "x"; }
			if ((_spring_array[idx]._index1 - StuffWorker::_element_list[a]._layer_springs[b]._index1 - mass_offset) != 0) { std::cout << "x"; }
			idx++;
		}
		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._time_springs.size(); b++)
		{
			if ((_spring_array[idx]._index0 - StuffWorker::_element_list[a]._time_springs[b]._index0 - mass_offset) != 0) { std::cout << "x"; }
			if ((_spring_array[idx]._index1 - StuffWorker::_element_list[a]._time_springs[b]._index1 - mass_offset) != 0) { std::cout << "x"; }
			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._auxiliary_springs.size(); b++)
		{
			if ((_spring_array[idx]._index0 - StuffWorker::_element_list[a]._auxiliary_springs[b]._index0 - mass_offset) != 0) { std::cout << "x"; }
			if ((_spring_array[idx]._index1 - StuffWorker::_element_list[a]._auxiliary_springs[b]._index1 - mass_offset) != 0) { std::cout << "x"; }
			idx++;
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._neg_space_springs.size(); b++)
		{
			if ((_spring_array[idx]._index0 - StuffWorker::_element_list[a]._neg_space_springs[b]._index0 - mass_offset) != 0) { std::cout << "x"; }
			if ((_spring_array[idx]._index1 - StuffWorker::_element_list[a]._neg_space_springs[b]._index1 - mass_offset) != 0) { std::cout << "x"; }
			idx++;
		}
	}*/

}

// call this only once
void CUDAWorker::InitSpringData()
{
	int idx = 0;
	for (unsigned int a = 0; a < StuffWorker::_element_list.size(); a++)
	{
		int m_offset = StuffWorker::_element_list[a]._cuda_mass_array_offset;

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._layer_springs.size(); b++)
		{
			AnIndexedLine ln = StuffWorker::_element_list[a]._layer_springs[b];
			_spring_array[idx++] = SpringGPU(ln._index0 + m_offset, 
				                             ln._index1 + m_offset, 
				                             ln._dist, 
				                             0); // 0
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._time_springs.size(); b++)
		{
			AnIndexedLine ln = StuffWorker::_element_list[a]._time_springs[b];
			_spring_array[idx++] = SpringGPU(ln._index0 + m_offset, 
				                             ln._index1 + m_offset, 
				                             ln._dist, 1); // 1
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._auxiliary_springs.size(); b++)
		{
			AnIndexedLine ln = StuffWorker::_element_list[a]._auxiliary_springs[b];
			_spring_array[idx++] = SpringGPU(ln._index0 + m_offset, 
				                             ln._index1 + m_offset, 
				                             ln._dist, 
				                             2); // 2
		}

		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._neg_space_springs.size(); b++)
		{
			AnIndexedLine ln = StuffWorker::_element_list[a]._neg_space_springs[b];
			_spring_array[idx++] = SpringGPU(ln._index0 + m_offset, 
				                             ln._index1 + m_offset, 
				                             ln._dist, 
				                             3); // 3
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
			_pos_array[idx]._x = StuffWorker::_element_list[a]._massList[b]._pos._x;
			_pos_array[idx]._y = StuffWorker::_element_list[a]._massList[b]._pos._y;
			_pos_array[idx]._z = StuffWorker::_element_list[a]._massList[b]._pos._z;
			idx++;
		}
	}
}

void CUDAWorker::SendPositionAndVelocityData()
{
	/*int idx = 0;
	for (unsigned int a = 0; a < StuffWorker::_element_list.size(); a++)
	{
		for (unsigned int b = 0; b < StuffWorker::_element_list[a]._massList.size(); b++)
		{
			CopyVector_CPU2GPU(StuffWorker::_element_list[a]._massList[b]._pos, &_pos_array[idx]);
			CopyVector_CPU2GPU(StuffWorker::_element_list[a]._massList[b]._velocity, &_velocity_array[idx]);
			idx++;
		}
	}*/
}

void CUDAWorker::SendForceData()
{
	/*int idx = 0;
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
	}*/

	// test, delete me
	/*idx = 0;
	for (unsigned int a = 0; a < _num_vertex; a++)
	{
		std::cout << _edge_force_array[a]._x << ", " << _edge_force_array[a]._y << ", " << _edge_force_array[a]._z << "\n";
	}

	std::cout << "done\n";*/
}

/*void CUDAWorker::CopyVector_CPU2GPU(const A3DVector& src, A3DVectorGPU* dest)
{
	dest->_x = src._x;
	dest->_y = src._y;
	dest->_z = src._z;
}*/

/*void CUDAWorker::CopyVector_GPU2CPU(A3DVectorGPU* src, A3DVector& dest)
{
	dest._x = src->_x;
	dest._y = src->_y;
	dest._z = src->_z;
}*/
