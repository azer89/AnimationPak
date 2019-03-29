
#include "StuffWorker.h"
#include "OpenCVWrapper.h"

// static items
std::vector<AnElement>  StuffWorker::_element_list = std::vector<AnElement>();
std::vector<CollisionGrid2D*>  StuffWorker::_c_grid_list = std::vector< CollisionGrid2D * >();

StuffWorker::StuffWorker()//: _elem(0)
{

}

StuffWorker::~StuffWorker()
{
	_element_list.clear();
	if (_c_grid_list.size() > 0)
	{
		for (int a = _c_grid_list.size(); a >= 0; a--)
		{
			delete _c_grid_list[a];
		}
		_c_grid_list.clear();
	}
	/*if (_elem)
	{
		delete _elem;
	}*/

	//Ogre::MaterialManager::getSingleton().remove("Examples/TransparentTest2");
}

void StuffWorker::InitElements(Ogre::SceneManager* scnMgr)
{
	
	/*_elem = new AnElement;
	_elem->CreateStarTube();
	_elem->ScaleXY(0.2);
	Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("Wow");
	_elem->InitMesh(scnMgr, pNode, "StarTube1", "Examples/TransparentTest2");*/
	
	
	{
		AnElement elem;
		elem.CreateStarTube(0);
		elem.ScaleXY(0.2);
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode0");
		elem.InitMesh(scnMgr, pNode, "StarTube0", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}
	
	{
		AnElement elem;
		elem.CreateStarTube(1);
		elem.ScaleXY(0.2);
		elem.TranslateXY(250, 200);
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode1");
		elem.InitMesh(scnMgr, pNode, "StarTube1", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube(2);
		elem.ScaleXY(0.2);
		elem.TranslateXY(250, 0);
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode2");
		elem.InitMesh(scnMgr, pNode, "StarTube2", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube(3);
		elem.ScaleXY(0.2);
		elem.TranslateXY(0, 350);
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode3");
		elem.InitMesh(scnMgr, pNode, "StarTube3", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube(4);
		elem.ScaleXY(0.2);
		elem.TranslateXY(400, 400);
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode4");
		elem.InitMesh(scnMgr, pNode, "StarTube4", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube(5);
		elem.ScaleXY(0.2);
		elem.TranslateXY(400, 0);
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode5");
		elem.InitMesh(scnMgr, pNode, "StarTube5", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	_c_grid_list.push_back(new CollisionGrid2D); // 0
	_c_grid_list.push_back(new CollisionGrid2D); // 1
	_c_grid_list.push_back(new CollisionGrid2D); // 2
	_c_grid_list.push_back(new CollisionGrid2D); // 3
	_c_grid_list.push_back(new CollisionGrid2D); // 4
	_c_grid_list.push_back(new CollisionGrid2D); // 5

	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			int c_grid_idx = _element_list[a]._massList[b]._debug_which_layer;
			A3DVector p1 = _element_list[a]._massList[b]._pos;
			
			// assign mass to grid
			_c_grid_list[c_grid_idx]->InsertAPoint(p1._x, p1._y, a, b);

			// assign grid to mass
			_element_list[a]._massList[b]._c_grid = _c_grid_list[c_grid_idx];
		}
	}
}

void StuffWorker::UpdateElements()
{
	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].UpdateBackend();
	}
}

