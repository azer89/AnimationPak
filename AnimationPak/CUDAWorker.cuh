

#ifndef CUDAWORKER_H
#define CUDAWORKER_H

#include "A3DVectorGPU.h"

class CUDAWorker
{
public:
	CUDAWorker();
	~CUDAWorker();

	void InitCUDA(int num_vertex);

	// forces
	/*A3DVectorGPU* _edge_force_array;
	A3DVectorGPU* _z_force_array;
	A3DVectorGPU* _repulsion_force_array;
	A3DVectorGPU* _boundary_force_array;
	A3DVectorGPU* _overlap_force_array;
	A3DVectorGPU* _rotation_force_array;*/

private:
	int TestCUDA();
};

#endif
