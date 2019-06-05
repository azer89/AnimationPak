

#include "ContainerWorker.h"
#include "PathIO.h"

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
	PathIO pathIO;
	_2d_container = pathIO.LoadElement(SystemParams::_container_file_name)[0];

	/*_2d_container.push_back(A2DVector(0,   0));
	_2d_container.push_back(A2DVector(0,   500));
	_2d_container.push_back(A2DVector(500, 500));
	_2d_container.push_back(A2DVector(500, 0));*/
}

void ContainerWorker::CreateOgreContainer(Ogre::SceneManager* scnMgr)
{
	
	float len = SystemParams::_upscaleFactor;

	// cube
	/*std::deque<Ogre::Vector3> cubePoints;
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
	}*/

	//In the initialization somewhere, create the initial lines object :
	// "Examples/BlueMat"
	Ogre::MaterialPtr material = Ogre::MaterialManager::getSingleton().getByName("Examples/BlueMat");
	DynamicLines * cube_lines = new DynamicLines(material, Ogre::RenderOperation::OT_LINE_LIST);
	/*for (int i = 0; i<cubePoints.size(); i++) 
	{
		cube_lines->addPoint(cubePoints[i]);
	}*/
	// ----- front ----- 
	for (int i = 0; i < _2d_container.size() - 1; i++)
	{
		cube_lines->addPoint(Ogre::Vector3(_2d_container[i].x, _2d_container[i].y, 0.0f));
		cube_lines->addPoint(Ogre::Vector3(_2d_container[i+1].x, _2d_container[i+1].y, 0.0f));
	}
	cube_lines->addPoint(Ogre::Vector3(_2d_container[_2d_container.size() - 1].x, _2d_container[_2d_container.size() - 1].y, 0.0f));
	cube_lines->addPoint(Ogre::Vector3(_2d_container[0].x, _2d_container[0].y, 0.0f));
	// ----- back ----- 
	for (int i = 0; i < _2d_container.size() - 1; i++)
	{
		cube_lines->addPoint(Ogre::Vector3(_2d_container[i].x, _2d_container[i].y, -len));
		cube_lines->addPoint(Ogre::Vector3(_2d_container[i + 1].x, _2d_container[i + 1].y, -len));
	}
	cube_lines->addPoint(Ogre::Vector3(_2d_container[_2d_container.size() - 1].x, _2d_container[_2d_container.size() - 1].y, -len));
	cube_lines->addPoint(Ogre::Vector3(_2d_container[0].x, _2d_container[0].y, -len));
	// ----- in between ----- 
	for (int i = 0; i < _2d_container.size(); i++)
	{
		cube_lines->addPoint(Ogre::Vector3(_2d_container[i].x, _2d_container[i].y, 0.0f));
		cube_lines->addPoint(Ogre::Vector3(_2d_container[i].x, _2d_container[i].y, -len));
	}

	cube_lines->update();
	Ogre::SceneNode *containerNode = scnMgr->getRootSceneNode()->createChildSceneNode("ContainerNode");
	containerNode->attachObject(cube_lines);

	
}