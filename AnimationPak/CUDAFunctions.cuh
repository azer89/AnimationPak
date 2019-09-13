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
	return A3DVectorGPU(p._x + v._x, 
		                p._y + v._y, 
		                p._z + v._z);
}

__device__
A3DVectorGPU operator-(const A3DVectorGPU& p, const A3DVectorGPU& v)
{
	return A3DVectorGPU(p._x - v._x, 
		                p._y - v._y, 
		                p._z - v._z);
}

__device__
A3DVectorGPU operator*(const A3DVectorGPU& p, const float& f)
{
	return A3DVectorGPU(p._x * f, 
		                p._y * f, 
		                p._z * f);
}

__device__
A3DVectorGPU operator/(const A3DVectorGPU& p, const float& f)
{
	return A3DVectorGPU(p._x / f, 
		                p._y / f, 
		                p._z / f);
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

__global__ void SolveForSprings3D_Linear_GPU(SpringGPU* spring_array,
	A3DVectorGPU* pos_array,
	A3DVectorGPU* edge_force_array,
	float* spring_diff_array, // debug delete me
	float* _spring_k_array, // debug delete me
	float* _spring_signval_array, // debug delete me
	float* _spring_mag_array, // debug delete me
	A3DVectorGPU* _spring_dir_array, // debug delete me
	float* spring_parameters,
	int n_springs)
{


	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n_springs; i += stride)
	{
		A3DVectorGPU dir;
		A3DVectorGPU dir_not_unit;
		A3DVectorGPU eForce;

		A3DVectorGPU temp1; // debug delete me
		A3DVectorGPU temp2; // debug delete me

		float dist = 0;
		float diff = 0;
		float k = 0;
		int idx0, idx1;
		int spring_type;

		// parameters
		spring_type = spring_array[i]._spring_type;
		k = spring_parameters[spring_type];

		// check this!
		idx0 = spring_array[i]._index0;
		idx1 = spring_array[i]._index1;

		dir_not_unit = pos_array[idx1] - pos_array[idx0];
		//GetUnitAndDist(dir_not_unit, dir, dist);
		dist = Length(dir_not_unit);
		dir = dir_not_unit / dist;

		diff = dist - spring_array[i]._dist;
		spring_diff_array[i] = diff;// debug delete me
		_spring_k_array[i] = k;// debug delete me
		_spring_mag_array[i] = dist; // debug delete me
		_spring_dir_array[i] = dir; // debug delete me

		// squared version

		_spring_signval_array[i] = 0; // debug delete me

		eForce = dir * k *  diff;

		temp1 = edge_force_array[idx0];
		temp2 = edge_force_array[idx1];

		edge_force_array[idx0] = temp1 + eForce;
		edge_force_array[idx1] = temp2 - eForce;
	}
}

__global__ void SolveForSprings3D_GPU(SpringGPU* spring_array, // INPUT
									  A3DVectorGPU* pos_array, // INPUT
									  A3DVectorGPU* edge_force_array_springs, // OUTPUT
									  float* spring_parameters, // INPUT
									  int n_springs) // INPUT
{
	

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n_springs; i += stride)
	{
		// parameters
		int spring_type = spring_array[i]._spring_type;
		float k = spring_parameters[spring_type];

		// check this!
		int idx0 = spring_array[i]._index0;
		int idx1 = spring_array[i]._index1;

		A3DVectorGPU dir;
		float dist = 0;
		A3DVectorGPU dir_not_unit = pos_array[idx1] - pos_array[idx0];
		GetUnitAndDist(dir_not_unit, dir, dist);
		//float dist = Length(dir_not_unit);
		//dir = dir_not_unit / dist;

		float diff = dist - spring_array[i]._dist;
		//spring_diff_array[i] = diff;// debug delete me
		//_spring_k_array[i] = k;// debug delete me
		//_spring_mag_array[i] = dist; // debug delete me
		//_spring_dir_array[i] = dir; // debug delete me

		// squared version
		float signVal = 1;
		if (diff < 0) { signVal = -1; }

		//_spring_signval_array[i] = signVal; // debug delete me

		edge_force_array_springs[i] = dir * (k *  diff * diff * signVal); // OUTPUT

		//temp1 = edge_force_array[idx0];
		//temp2 = edge_force_array[idx1];

		//edge_force_array[idx0] = temp1 + eForce;
		//edge_force_array[idx1] = temp2 - eForce;
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
