

#include "ContainerWorker.h"
#include "PathIO.h"
#include "UtilityFunctions.h"

// library
#include "PoissonGenerator.h"

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

	int numPoints = SystemParams::_num_element_density;
	DefaultPRNG PRNG(SystemParams::_seed);
	PoissonGenerator pg;
	const auto points = pg.GeneratePoissonPoints(numPoints, PRNG);

	// ---------- iterate points ----------
	float offst = std::sqrt(2.0f) * SystemParams::_upscaleFactor;
	float shiftOffst = 0.5f * (offst - SystemParams::_upscaleFactor);
	for (auto i = points.begin(); i != points.end(); i++)
	{
		A2DVector pt((i->x * offst) - shiftOffst, (i->y * offst) - shiftOffst);

		bool isInside = UtilityFunctions::InsidePolygon(_2d_container, pt.x, pt.y); 

		if (isInside)
		{
			_randomPositions.push_back(pt);
		}

	}
	int num_pos_limit = SystemParams::_num_element_pos_limit;
	if (num_pos_limit < _randomPositions.size())
	{
		while (_randomPositions.size() != num_pos_limit)
		{
			std::mt19937 g(SystemParams::_seed);
			std::shuffle(_randomPositions.begin(), _randomPositions.end(), g);
			_randomPositions.erase(_randomPositions.begin());
		}
	}
}

void ContainerWorker::UpdateOgre3D()
{
	_cube_node->setVisible(SystemParams::_show_container);
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
	_cube_lines = new DynamicLines(material, Ogre::RenderOperation::OT_LINE_LIST);
	/*for (int i = 0; i<cubePoints.size(); i++) 
	{
		cube_lines->addPoint(cubePoints[i]);
	}*/
	// ----- front ----- 
	for (int i = 0; i < _2d_container.size() - 1; i++)
	{
		_cube_lines->addPoint(Ogre::Vector3(_2d_container[i].x, _2d_container[i].y, 0.0f));
		_cube_lines->addPoint(Ogre::Vector3(_2d_container[i+1].x, _2d_container[i+1].y, 0.0f));
	}
	_cube_lines->addPoint(Ogre::Vector3(_2d_container[_2d_container.size() - 1].x, _2d_container[_2d_container.size() - 1].y, 0.0f));
	_cube_lines->addPoint(Ogre::Vector3(_2d_container[0].x, _2d_container[0].y, 0.0f));
	// ----- back ----- 
	for (int i = 0; i < _2d_container.size() - 1; i++)
	{
		_cube_lines->addPoint(Ogre::Vector3(_2d_container[i].x, _2d_container[i].y, -len));
		_cube_lines->addPoint(Ogre::Vector3(_2d_container[i + 1].x, _2d_container[i + 1].y, -len));
	}
	_cube_lines->addPoint(Ogre::Vector3(_2d_container[_2d_container.size() - 1].x, _2d_container[_2d_container.size() - 1].y, -len));
	_cube_lines->addPoint(Ogre::Vector3(_2d_container[0].x, _2d_container[0].y, -len));
	// ----- in between ----- 
	for (int i = 0; i < _2d_container.size(); i++)
	{
		_cube_lines->addPoint(Ogre::Vector3(_2d_container[i].x, _2d_container[i].y, 0.0f));
		_cube_lines->addPoint(Ogre::Vector3(_2d_container[i].x, _2d_container[i].y, -len));
	}

	_cube_lines->update();
	_cube_node = scnMgr->getRootSceneNode()->createChildSceneNode("ContainerNode");
	_cube_node->attachObject(_cube_lines);

	
}