#ifndef CONTAINER_WORKER_H
#define CONTAINER_WORKER_H

#include "A2DVector.h"
#include "A3DVector.h"

// ogre
#include <Ogre.h>
#include <OgreApplicationContext.h>
#include <OgrePrerequisites.h>
#include <OgreMaterial.h>
#include <OgreMesh.h>
#include <OgreSceneManager.h>

#include "DynamicLines.h"

class ContainerWorker
{
public:
	ContainerWorker();
	~ContainerWorker();

	void LoadContainer();

	void CreateOgreContainer(Ogre::SceneManager* scnMgr);

	void UpdateOgre3D();

	std::vector<A2DVector> _2d_container;

	std::vector<A2DVector> _randomPositions;  // for elements

private:
	DynamicLines* _container_3d_lines;
	Ogre::SceneNode* _container_3d_node;

	DynamicLines* _layer_container_3d_lines;
	Ogre::SceneNode* _layer_container_3d_node;
};

#endif