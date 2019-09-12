


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef CUDAWORKER_H
#define CUDAWORKER_H

#include "A3DVector.h"
#include "A3DVectorGPU.h"
#include "AnElement.h"
#include "AMass.h"

#include "SpringGPU.h"

class CUDAWorker
{
public:
	CUDAWorker();
	~CUDAWorker();

	void InitCUDA(int num_vertex, int num_spring);

	void SendPositionData();
	void SendPositionAndVelocityData();
	void SendForceData(); 
	void SendSpringLengths();

	void RetrievePositionAndVelocityData();
	void RetrieveEdgeForceData();
	void UnitTestSpringGPU();
	
	void InitSpringData();
	
	void ResetForces();
	void SolveForSprings3D();
	void Simulate(float dt, float velocity_cap);         // integrate
	
	// forces
	A3DVectorGPU* _edge_force_array;
	A3DVectorGPU* _z_force_array;
	A3DVectorGPU* _repulsion_force_array;
	A3DVectorGPU* _boundary_force_array;
	A3DVectorGPU* _overlap_force_array;
	A3DVectorGPU* _rotation_force_array;

	// mass positions
	A3DVectorGPU* _pos_array;

	// mass velocities
	A3DVectorGPU* _velocity_array;

	
	float* _spring_diff_array;       // debug delete me
	float* _spring_k_array;          // debug delete me
	float* _spring_signval_array;    // debug delete me
	float* _spring_mag_array;        // debug delete me
	A3DVectorGPU* _spring_dir_array; // debug delete me

	// springs
	SpringGPU* _spring_array;
	float* _spring_parameters;

	// tool
	void CopyVector_CPU2GPU(const A3DVector& src, A3DVectorGPU* dest);
	void CopyVector_GPU2CPU(A3DVectorGPU* src, A3DVector& dest);

private:
	int _num_vertex; // how many vertices in the system
	int _num_spring; // how many springs in the system
};

#endif
