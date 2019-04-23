
#include "StuffWorker.h"
#include "OpenCVWrapper.h"
#include "TetWrapper.h"
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
	float initialScale = 0.05; // 0.05

	/*{
		int idx = _element_list.size();
		AnElement elem;
		elem.CreateStarTube(idx);
		elem.ScaleXY(initialScale);
		elem.TranslateXY(10, 10);
		//A2DVector startPt(25, 25, 0);
		//A2DVector endPt(475, 475, 0);
		//elem.AdjustEnds( A2DVector(75, 75), A2DVector(520, 520) );
		elem.AdjustEndPosition(A2DVector(480, 480));
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
		elem.InitMeshOgre3D(scnMgr, pNode, "StarTube" + std::to_string(idx), "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}*/

	
	/*{
	// don't use this
		AnElement elem;
		elem.CreateStarTube(1);
		elem.ScaleXY(initialScale);
		elem.TranslateXY(450, 450);
		elem.AdjustEnds(A2DVector(475, 475), A2DVector(35, 35));
		//elem.TranslateXY(250, 400);
		//elem.ResetSpringRestLengths();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode1");
		elem.InitMesh(scnMgr, pNode, "StarTube1", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}*/

	std::vector<A2DVector> posArray;

	posArray.push_back(A2DVector(250, 0));
	posArray.push_back(A2DVector(0, 350));
	posArray.push_back(A2DVector(100, 400));
	posArray.push_back(A2DVector(400, 0));
	posArray.push_back(A2DVector(40, 240));
	posArray.push_back(A2DVector(330, 140));
	posArray.push_back(A2DVector(400, 250));
	posArray.push_back(A2DVector(0, 180));
	posArray.push_back(A2DVector(170, 210));
	posArray.push_back(A2DVector(320, 280));
	posArray.push_back(A2DVector(350, 280));
	posArray.push_back(A2DVector(350, 220));

	for (int a = 0; a < posArray.size(); a++)
	{
		int idx = _element_list.size();
		AnElement elem;
		//elem.CreateStarTube(idx);
		elem.Triangularization(idx);
		elem.ScaleXY(initialScale);
		elem.TranslateXY(posArray[a].x, posArray[a].y);		
		//elem.ResetSpringRestLengths();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
		elem.InitMeshOgre3D(scnMgr, pNode, "StarTube" + std::to_string(idx), "Examples/TransparentTest2");
		_element_list.push_back(elem);
			
	}
	
	//{
	//	int idx = _element_list.size();
	//	AnElement elem;
	//	elem.CreateStarTube(idx);
	//	elem.ScaleXY(initialScale);
	//	elem.TranslateXY(250, 0);
	//	//elem.ResetSpringRestLengths();
	//	Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + idx);
	//	elem.InitMesh(scnMgr, pNode, "StarTube" + idx, "Examples/TransparentTest2");
	//	_element_list.push_back(elem);
	//}

	//
	//{
	//	int idx = _element_list.size();
	//	AnElement elem;
	//	elem.CreateStarTube(idx);
	//	elem.ScaleXY(initialScale);
	//	elem.TranslateXY(0, 350);
	//	//elem.ResetSpringRestLengths();
	//	Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + idx);
	//	elem.InitMesh(scnMgr, pNode, "StarTube" + idx, "Examples/TransparentTest2");
	//	_element_list.push_back(elem);
	//}
	//
	//{
	//	int idx = _element_list.size();
	//	AnElement elem;
	//	elem.CreateStarTube(idx);
	//	elem.ScaleXY(initialScale);
	//	elem.TranslateXY(100, 400);
	//	//elem.ResetSpringRestLengths();
	//	Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + idx);
	//	elem.InitMesh(scnMgr, pNode, "StarTube" + idx, "Examples/TransparentTest2");
	//	_element_list.push_back(elem);
	//}
	//
	//{
	//	int idx = _element_list.size();
	//	AnElement elem;
	//	elem.CreateStarTube(idx);
	//	elem.ScaleXY(initialScale);
	//	elem.TranslateXY(400, 0);
	//	//elem.ResetSpringRestLengths();
	//	Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + idx);
	//	elem.InitMesh(scnMgr, pNode, "StarTube" + idx, "Examples/TransparentTest2");
	//	_element_list.push_back(elem);
	//}
	//
	//{
	//	int idx = _element_list.size();
	//	AnElement elem;
	//	elem.CreateStarTube(idx);
	//	elem.ScaleXY(initialScale);
	//	elem.TranslateXY(40, 240);
	//	//elem.ResetSpringRestLengths();
	//	Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + idx);
	//	elem.InitMesh(scnMgr, pNode, "StarTube" + idx, "Examples/TransparentTest2");
	//	_element_list.push_back(elem);
	//}
	//
	//{
	//	int idx = _element_list.size();
	//	AnElement elem;
	//	elem.CreateStarTube(idx);
	//	elem.ScaleXY(initialScale);
	//	elem.TranslateXY(330, 140);
	//	//elem.ResetSpringRestLengths();
	//	Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + idx);
	//	elem.InitMesh(scnMgr, pNode, "StarTube" + idx, "Examples/TransparentTest2");
	//	_element_list.push_back(elem);
	//}
	//
	//{
	//	int idx = _element_list.size();
	//	AnElement elem;
	//	elem.CreateStarTube(idx);
	//	elem.ScaleXY(initialScale);
	//	elem.TranslateXY(400, 250);
	//	//elem.ResetSpringRestLengths();
	//	Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + idx);
	//	elem.InitMesh(scnMgr, pNode, "StarTube" + idx, "Examples/TransparentTest2");
	//	_element_list.push_back(elem);
	//}
	//
	//{
	//	int idx = _element_list.size();
	//	AnElement elem;
	//	elem.CreateStarTube(idx);
	//	elem.ScaleXY(initialScale);
	//	elem.TranslateXY(0, 180);
	//	//elem.ResetSpringRestLengths();
	//	Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + idx);
	//	elem.InitMesh(scnMgr, pNode, "StarTube" + idx, "Examples/TransparentTest2");
	//	_element_list.push_back(elem);
	//}

	//{
	//	int idx = _element_list.size();
	//	AnElement elem;
	//	elem.CreateStarTube(idx);
	//	elem.ScaleXY(initialScale);
	//	elem.TranslateXY(170, 200);
	//	//elem.ResetSpringRestLengths();
	//	Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + idx);
	//	elem.InitMesh(scnMgr, pNode, "StarTube" + idx, "Examples/TransparentTest2");
	//	_element_list.push_back(elem);
	//}

	//{
	//	int idx = _element_list.size();
	//	AnElement elem;
	//	elem.CreateStarTube(idx);
	//	elem.ScaleXY(initialScale);
	//	elem.TranslateXY(320, 270);
	//	//elem.ResetSpringRestLengths();
	//	Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + idx);
	//	elem.InitMesh(scnMgr, pNode, "StarTube" + idx, "Examples/TransparentTest2");
	//	_element_list.push_back(elem);
	//}


	//{
	//	int idx = _element_list.size();
	//	AnElement elem;
	//	elem.CreateStarTube(idx);
	//	elem.ScaleXY(initialScale);
	//	elem.TranslateXY(350, 270);
	//	//elem.ResetSpringRestLengths();
	//	Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + idx);
	//	elem.InitMesh(scnMgr, pNode, "StarTube" + idx, "Examples/TransparentTest2");
	//	_element_list.push_back(elem);
	//}

	//{
	//	int idx = _element_list.size();
	//	AnElement elem;
	//	elem.CreateStarTube(idx);
	//	elem.ScaleXY(initialScale);
	//	elem.TranslateXY(350, 220);
	//	//elem.ResetSpringRestLengths();
	//	Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + idx);
	//	elem.InitMesh(scnMgr, pNode, "StarTube" + idx, "Examples/TransparentTest2");
	//	_element_list.push_back(elem);
	//}

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
			int c_grid_idx = _element_list[a]._massList[b]._layer_idx;
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
	/*for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].UpdateBackend();
	}*/

	// update collision grid
	std::vector<int> iters; // TODO can be better
	for (int a = 0; a < _c_grid_list.size(); a++) { iters.push_back(0); }

	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{

			int c_grid_idx = _element_list[a]._massList[b]._layer_idx;
			int layer_iter = iters[c_grid_idx];
			A3DVector p1 = _element_list[a]._massList[b]._pos;

			// update pt
			//_c_grid_list[c_grid_idx]->InsertAPoint(p1._x, p1._y, a, b);
			_c_grid_list[c_grid_idx]->_objects[layer_iter]->_x = p1._x;
			_c_grid_list[c_grid_idx]->_objects[layer_iter]->_y = p1._y;

			iters[c_grid_idx]++; // increment
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

	// grow
	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].Grow(SystemParams::_growth_scale_iter, SystemParams::_dt);
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

void StuffWorker::UpdateOgre3D()
{
	for (int a = 0; a < _element_list.size(); a++)
	{
		//_element_list[a].UpdateMeshOgre3D();
		_element_list[a].UpdateSpringDisplayOgre3D();
		//_element_list[a].UpdateClosestPtsDisplayOgre3D();
	}
}

void StuffWorker::SaveFrames()
{
	std::cout << "please uncomment me\n";
	/*AVideoCreator vCreator;
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
				int layerIdx = _element_list[a]._massList[ln._index0]._layer_idx;
				vCreator.DrawLine(pt1, pt2, layerIdx);
			}
		}
	}

	std::stringstream ss;
	ss << SystemParams::_save_folder << "PNG\\";
	vCreator.Save(ss.str());*/
}
