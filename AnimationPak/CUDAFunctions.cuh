#ifndef __CUDA_FUNCTIONS_H__
#define __CUDA_FUNCTIONS_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "A3DVector.h"
#include "A3DVectorGPU.h"
#include "AnElement.h"
#include "AMass.h"

#include "SpringGPU.h"

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

	//if (vlength == 0) { return A3DVectorGPU(0, 0, 0); }

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
	//A3DVectorGPU pt0;
	//A3DVectorGPU pt1;
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

		dir_not_unit = DirectionTo(pos_array[idx0], pos_array[idx1]);
		GetUnitAndDist(dir_not_unit, dir, dist);

		diff = dist - spring_array[i]._dist;

		// squared version
		signVal = 1;
		if (diff < 0) { signVal = -1; }

		eForce = dir * (k *  diff * diff * signVal);

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

#endif
