#ifndef CONTAINER_WORKER_H
#define CONTAINER_WORKER_H

#include "A2DVector.h"
#include "A3DVector.h"

class ContainerWorker
{
public:
	ContainerWorker();
	~ContainerWorker();

	void LoadContainer();

	std::vector<A2DVector> _2d_container;
};

#endif