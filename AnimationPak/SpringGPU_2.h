#ifndef SPRING_GPU_2_H
#define SPRING_GPU_2_H

#include "cuda_runtime.h"

struct SpringGPU_2
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

	__host__ __device__ SpringGPU_2()
	{
		_index0 = -1;
		_index1 = -1;
		_dist = 0;
		_spring_type = -1;
		//_scale = 1;
	}

	__host__ __device__ SpringGPU_2(int index0, int index1, float dist, int spring_type)
	{
		_index0 = index0;
		_index1 = index1;
		_dist = dist;
		_spring_type = spring_type;
		//_scale = 1;
	}
};

#endif
