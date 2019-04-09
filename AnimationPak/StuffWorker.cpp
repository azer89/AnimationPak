
#include "StuffWorker.h"
#include "OpenCVWrapper.h"
#include "AVideoCreator.h"

#include "ContainerWorker.h"

// static items
std::vector<AnElement>  StuffWorker::_element_list = std::vector<AnElement>();
std::vector<CollisionGrid2D*>  StuffWorker::_c_grid_list = std::vector< CollisionGrid2D * >();

StuffWorker::StuffWorker() : _containerWorker(0)//: _elem(0)
{
	_containerWorker = new ContainerWorker;
	_containerWorker->LoadContainer();
}

StuffWorker::~StuffWorker()
{
	if (_containerWorker) { delete _containerWorker; }

	_element_list.clear();
	if (_c_grid_list.size() > 0)
	{
		for (int a = _c_grid_list.size() - 1; a >= 0; a--)
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

	{
		AnElement elem;
		elem.CreateStarTube(0);
		elem.ScaleXY(0.15);
		//elem.ResetSpringRestLengths();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode0");
		elem.InitMesh(scnMgr, pNode, "StarTube0", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube(1);
		elem.ScaleXY(0.29);
		elem.TranslateXY(250, 400);
		//elem.ResetSpringRestLengths();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode1");
		elem.InitMesh(scnMgr, pNode, "StarTube1", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube(2);
		elem.ScaleXY(0.21);
		elem.TranslateXY(250, 0);
		//elem.ResetSpringRestLengths();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode2");
		elem.InitMesh(scnMgr, pNode, "StarTube2", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube(3);
		elem.ScaleXY(0.2);
		elem.TranslateXY(0, 350);
		//elem.ResetSpringRestLengths();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode3");
		elem.InitMesh(scnMgr, pNode, "StarTube3", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube(4);
		elem.ScaleXY(0.17);
		elem.TranslateXY(400, 400);
		//elem.ResetSpringRestLengths();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode4");
		elem.InitMesh(scnMgr, pNode, "StarTube4", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube(5);
		elem.ScaleXY(0.22);
		elem.TranslateXY(400, 0);
		//elem.ResetSpringRestLengths();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode5");
		elem.InitMesh(scnMgr, pNode, "StarTube5", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube(6);
		elem.ScaleXY(0.19);
		elem.TranslateXY(240, 240);
		//elem.ResetSpringRestLengths();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode6");
		elem.InitMesh(scnMgr, pNode, "StarTube6", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube(7);
		elem.ScaleXY(0.22);
		elem.TranslateXY(130, 140);
		//elem.ResetSpringRestLengths();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode7");
		elem.InitMesh(scnMgr, pNode, "StarTube7", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube(8);
		elem.ScaleXY(0.2);
		elem.TranslateXY(400, 250);
		//elem.ResetSpringRestLengths();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode8");
		elem.InitMesh(scnMgr, pNode, "StarTube8", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube(9);
		elem.ScaleXY(0.2);
		elem.TranslateXY(0, 180);
		//elem.ResetSpringRestLengths();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode9");
		elem.InitMesh(scnMgr, pNode, "StarTube9", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	for(int a = 0; a < SystemParams::_num_layer; a++)
	{
		_c_grid_list.push_back(new CollisionGrid2D); // 0
	}
	//_c_grid_list.push_back(new CollisionGrid2D); // 1
	//_c_grid_list.push_back(new CollisionGrid2D); // 2
	//_c_grid_list.push_back(new CollisionGrid2D); // 3
	//_c_grid_list.push_back(new CollisionGrid2D); // 4
	//_c_grid_list.push_back(new CollisionGrid2D); // 5

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

void StuffWorker::Update()
{
	// ???
	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].UpdateBackend();
	}

	// update collision grid
	std::vector<int> iters; // TODO can be better
	for (int a = 0; a < _c_grid_list.size(); a++) { iters.push_back(0); }

	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{

			int c_grid_idx = _element_list[a]._massList[b]._debug_which_layer;
			int layer_iter = iters[c_grid_idx];
			A3DVector p1 = _element_list[a]._massList[b]._pos;

			// update pt
			//_c_grid_list[c_grid_idx]->InsertAPoint(p1._x, p1._y, a, b);
			_c_grid_list[c_grid_idx]->_objects[layer_iter]->_x = p1._x;
			_c_grid_list[c_grid_idx]->_objects[layer_iter]->_y = p1._y;

			iters[c_grid_idx] += 1; // increment
		}
	}
	for (int a = 0; a < _c_grid_list.size(); a++)
	{
		_c_grid_list[a]->MovePoints();
		_c_grid_list[a]->PrecomputeGraphIndices();
	}

	// update closest points
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_element_list[a]._massList[b].GetClosestPoint();
		}

	}

}

void StuffWorker::Reset()
{
	// update closest points
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_element_list[a]._massList[b].Init();
		}

	}
}

void StuffWorker::Solve()
{
	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].SolveForSprings();

		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_element_list[a]._massList[b].Solve(_containerWorker->_2d_container);
		}
	}
}

void StuffWorker::Simulate()
{
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_element_list[a]._massList[b].Simulate(SystemParams::_dt);
		}
	}
}

void StuffWorker::ImposeConstraints()
{
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_element_list[a]._massList[b].ImposeConstraints();
		}
	}
}

void StuffWorker::UpdateViz()
{
	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].UpdateMesh2();
	}
}

void StuffWorker::SaveFrames()
{
	AVideoCreator vCreator;
	vCreator.Init();
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._triEdges.size(); b++)
		{
			AnIndexedLine ln = _element_list[a]._triEdges[b];
			if (!ln._isLayer2Layer)
			{
				A2DVector pt1 = _element_list[a]._massList[ln._index0]._pos.GetA2DVector();
				A2DVector pt2 = _element_list[a]._massList[ln._index1]._pos.GetA2DVector();
				int layerIdx = _element_list[a]._massList[ln._index0]._debug_which_layer;
				vCreator.DrawLine(pt1, pt2, layerIdx);
			}
		}
	}

	std::stringstream ss;
	ss << SystemParams::_save_folder << "PNG\\";
	vCreator.Save(ss.str());
}
