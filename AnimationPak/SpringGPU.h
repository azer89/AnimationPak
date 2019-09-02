#ifndef SPRING_GPU_H
#define SPRING_GPU_H

#include "cuda_runtime.h"

struct SpringGPU
{
public:
	
	int   _index0;
	int   _index1; 
	float _dist;

	__host__ __device__ SpringGPU()
	{
		_index0 = -1;
		_index1 = -1;
		_dist = 0;
	}

	__host__ __device__ SpringGPU(int index0, int index1, float dist)
	{
		_index0 = index0;
		_index1 = index1;
		_dist = dist;
	}
};

#endif
