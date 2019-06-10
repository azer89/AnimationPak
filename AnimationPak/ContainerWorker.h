#ifndef CONTAINER_WORKER_H
#define CONTAINER_WORKER_H

#include "A2DVector.h"
#include "A3DVector.h"

// ogre

#include <OgreSceneManager.h>

class ContainerWorker
{
public:
	ContainerWorker();
	~ContainerWorker();

	void LoadContainer();

	void CreateOgreContainer(Ogre::SceneManager* scnMgr);

	std::vector<A2DVector> _2d_container;

	std::vector<A2DVector> _randomPositions;  // for elements
};

#endif