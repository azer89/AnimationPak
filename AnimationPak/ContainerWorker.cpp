

#include "ContainerWorker.h"

ContainerWorker::ContainerWorker()
{

}

ContainerWorker::~ContainerWorker()
{
}

void ContainerWorker::LoadContainer()
{
	_2d_container.push_back(A2DVector(0,   0));
	_2d_container.push_back(A2DVector(0,   500));
	_2d_container.push_back(A2DVector(500, 500));
	_2d_container.push_back(A2DVector(500, 0));
}