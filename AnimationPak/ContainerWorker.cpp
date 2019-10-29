

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
		float dist = UtilityFunctions::DistanceToClosedCurve(_2d_container, pt);

		//std::cout << dist << "\n";

		if (isInside && dist > 30.0)
		{
			_randomPositions.push_back(pt);
		}

	}

	/*int num_pos_limit = SystemParams::_num_element_pos_limit;
	if (num_pos_limit < _randomPositions.size())
	{
		while (_randomPositions.size() != num_pos_limit)
		{
			std::mt19937 g(SystemParams::_seed);
			std::shuffle(_randomPositions.begin(), _randomPositions.end(), g);
			_randomPositions.erase(_randomPositions.begin());
		}
	}*/
}

void ContainerWorker::UpdateOgre3D()
{
	if (SystemParams::_show_container)
	{
		
		if (SystemParams::_layer_slider_int == -1)
		{
			_container_3d_node->setVisible(true);
			_layer_container_3d_node->setVisible(false);
		}
		else
		{
			_container_3d_node->setVisible(false);
			_layer_container_3d_node->setVisible(true);

			_layer_container_3d_lines->clear();

			float z_pos = -((float)SystemParams::_layer_slider_int) * (((float)SystemParams::_upscaleFactor) / ((float)SystemParams::_num_layer));
			for (int i = 0; i < _2d_container.size() - 1; i++)
			{
				_layer_container_3d_lines->addPoint(Ogre::Vector3(_2d_container[i].x, _2d_container[i].y, z_pos));
				_layer_container_3d_lines->addPoint(Ogre::Vector3(_2d_container[i + 1].x, _2d_container[i + 1].y, z_pos));
			}
			_layer_container_3d_lines->addPoint(Ogre::Vector3(_2d_container[_2d_container.size() - 1].x, _2d_container[_2d_container.size() - 1].y, z_pos));
			_layer_container_3d_lines->addPoint(Ogre::Vector3(_2d_container[0].x, _2d_container[0].y, z_pos));

			_layer_container_3d_lines->update();
		}
	}
	else
	{
		_container_3d_node->setVisible(false);
		_layer_container_3d_node->setVisible(false);
	}
}

void ContainerWorker::CreateOgreContainer(Ogre::SceneManager* scnMgr)
{
	
	float len = SystemParams::_upscaleFactor;


	//In the initialization somewhere, create the initial lines object :
	// "Examples/BlueMat"
	Ogre::MaterialPtr container_mat = Ogre::MaterialManager::getSingleton().getByName("Examples/BlueMat");
	_container_3d_lines = new DynamicLines(container_mat, Ogre::RenderOperation::OT_LINE_LIST);
	/*for (int i = 0; i<cubePoints.size(); i++) 
	{
		cube_lines->addPoint(cubePoints[i]);
	}*/
	// ----- front ----- 
	for (int i = 0; i < _2d_container.size() - 1; i++)
	{
		_container_3d_lines->addPoint(Ogre::Vector3(_2d_container[i].x, _2d_container[i].y, 0.0f));
		_container_3d_lines->addPoint(Ogre::Vector3(_2d_container[i+1].x, _2d_container[i+1].y, 0.0f));
	}
	_container_3d_lines->addPoint(Ogre::Vector3(_2d_container[_2d_container.size() - 1].x, _2d_container[_2d_container.size() - 1].y, 0.0f));
	_container_3d_lines->addPoint(Ogre::Vector3(_2d_container[0].x, _2d_container[0].y, 0.0f));
	// ----- back ----- 
	for (int i = 0; i < _2d_container.size() - 1; i++)
	{
		_container_3d_lines->addPoint(Ogre::Vector3(_2d_container[i].x, _2d_container[i].y, -len));
		_container_3d_lines->addPoint(Ogre::Vector3(_2d_container[i + 1].x, _2d_container[i + 1].y, -len));
	}
	_container_3d_lines->addPoint(Ogre::Vector3(_2d_container[_2d_container.size() - 1].x, _2d_container[_2d_container.size() - 1].y, -len));
	_container_3d_lines->addPoint(Ogre::Vector3(_2d_container[0].x, _2d_container[0].y, -len));
	// ----- in between ----- 
	for (int i = 0; i < _2d_container.size(); i++)
	{
		_container_3d_lines->addPoint(Ogre::Vector3(_2d_container[i].x, _2d_container[i].y, 0.0f));
		_container_3d_lines->addPoint(Ogre::Vector3(_2d_container[i].x, _2d_container[i].y, -len));
	}

	_container_3d_lines->update();
	_container_3d_node = scnMgr->getRootSceneNode()->createChildSceneNode("Container_Node");
	_container_3d_node->attachObject(_container_3d_lines);


	// layer
	_layer_container_3d_lines = new DynamicLines(container_mat, Ogre::RenderOperation::OT_LINE_LIST);
	_layer_container_3d_lines->update();
	_layer_container_3d_node = scnMgr->getRootSceneNode()->createChildSceneNode("Layer_Container_Node");
	_layer_container_3d_node->attachObject(_layer_container_3d_lines);
	
}