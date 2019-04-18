

#include "ContainerWorker.h"

// ogre
#include <Ogre.h>
#include <OgreApplicationContext.h>
#include <OgrePrerequisites.h>
#include <OgreMaterial.h>
#include <OgreMesh.h>

#include "DynamicLines.h"

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

void ContainerWorker::CreateOgreContainer(Ogre::SceneManager* scnMgr)
{
	std::deque<Ogre::Vector3> cubePoints;

	float len = SystemParams::_upscaleFactor;

	// cube
	{
		// front
		cubePoints.push_back(Ogre::Vector3(0.0f, 0.0f, 0.0f));
		cubePoints.push_back(Ogre::Vector3(len, 0.0f, 0.0f));

		cubePoints.push_back(Ogre::Vector3(len, 0.0f, 0.0f));
		cubePoints.push_back(Ogre::Vector3(500.0f, len, 0.0f));

		cubePoints.push_back(Ogre::Vector3(len, len, 0.0f));
		cubePoints.push_back(Ogre::Vector3(0.0f, len, 0.0f));

		cubePoints.push_back(Ogre::Vector3(0.0f, len, 0.0f));
		cubePoints.push_back(Ogre::Vector3(0.0f, 0.0f, 0.0f));

		// back
		cubePoints.push_back(Ogre::Vector3(0.0f, 0.0f, -len));
		cubePoints.push_back(Ogre::Vector3(len, 0.0f, -len));

		cubePoints.push_back(Ogre::Vector3(len, 0.0f, -len));
		cubePoints.push_back(Ogre::Vector3(len, len, -len));

		cubePoints.push_back(Ogre::Vector3(len, len, -len));
		cubePoints.push_back(Ogre::Vector3(0.0f, len, -len));

		cubePoints.push_back(Ogre::Vector3(0.0f, len, -len));
		cubePoints.push_back(Ogre::Vector3(0.0f, 0.0f, -len));

		// left
		cubePoints.push_back(Ogre::Vector3(0.0f, len, 0.0f));
		cubePoints.push_back(Ogre::Vector3(0.0f, len, -len));

		cubePoints.push_back(Ogre::Vector3(0.0f, 0.0f, 0.0f));
		cubePoints.push_back(Ogre::Vector3(0.0f, 0.0f, -len));

		// right
		cubePoints.push_back(Ogre::Vector3(len, len, 0.0f));
		cubePoints.push_back(Ogre::Vector3(len, len, -len));

		cubePoints.push_back(Ogre::Vector3(len, 0.0f, 0.0f));
		cubePoints.push_back(Ogre::Vector3(len, 0.0f, -len));
	}

	//In the initialization somewhere, create the initial lines object :
	// "Examples/BlueMat"
	Ogre::MaterialPtr material = Ogre::MaterialManager::getSingleton().getByName("Examples/BlueMat");
	DynamicLines * cube_lines = new DynamicLines(material, Ogre::RenderOperation::OT_LINE_LIST);
	for (int i = 0; i<cubePoints.size(); i++) {
		cube_lines->addPoint(cubePoints[i]);
	}

	cube_lines->update();
	Ogre::SceneNode *containerNode = scnMgr->getRootSceneNode()->createChildSceneNode("ContainerNode");
	containerNode->attachObject(cube_lines);

	
}