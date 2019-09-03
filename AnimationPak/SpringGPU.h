#ifndef SPRING_GPU_H
#define SPRING_GPU_H

#include "cuda_runtime.h"

struct SpringGPU
{
public:
	
	int   _index0;
	int   _index1; 
	float _dist;

	/*
	0 layer springs
	1 time springs
	2 auxiliary springs
	3 negative space springs	
	*/
	int  _spring_type;

	__host__ __device__ SpringGPU()
	{
		_index0 = -1;
		_index1 = -1;
		_dist = 0;
		_spring_type = -1;
	}

	__host__ __device__ SpringGPU(int index0, int index1, float dist, int spring_type)
	{
		_index0 = index0;
		_index1 = index1;
		_dist = dist;
		_spring_type = spring_type;
	}
};

#endif
