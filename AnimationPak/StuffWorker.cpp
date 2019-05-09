
#include "StuffWorker.h"
#include "OpenCVWrapper.h"
#include "TetWrapper.h"
#include "AVideoCreator.h"

#include "ContainerWorker.h"

// static variables
std::vector<AnElement>  StuffWorker::_element_list = std::vector<AnElement>();
std::vector<CollisionGrid2D*>  StuffWorker::_c_grid_list = std::vector< CollisionGrid2D * >();

// static variables for interpolation
//bool  StuffWorker::_interpolation_mode = false;
//int   StuffWorker::_interpolation_iter = 0;
//float StuffWorker::_interpolation_value = 0;

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
	float initialScale = SystemParams::_element_initial_scale; // 0.05

	{
		int idx = _element_list.size();
		AnElement elem;
		elem.Triangularization(idx);
		elem.ScaleXY(initialScale);
		elem.TranslateXY(20, 20);
		elem.AdjustEndPosition(A2DVector(490, 490));
		elem.CalculateRestStructure();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
		elem.InitMeshOgre3D(scnMgr, pNode, "StarTube" + std::to_string(idx), "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	
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

	//posArray.push_back(A2DVector(50, 270));
	//posArray.push_back(A2DVector(450, 220));

	posArray.push_back(A2DVector(250, 0));
	posArray.push_back(A2DVector(0, 350));
	posArray.push_back(A2DVector(100, 400));
	posArray.push_back(A2DVector(400, 0));
	posArray.push_back(A2DVector(400, 30)); //
	posArray.push_back(A2DVector(400, 70)); //
	posArray.push_back(A2DVector(40, 240));
	posArray.push_back(A2DVector(330, 140));
	posArray.push_back(A2DVector(400, 250));
	posArray.push_back(A2DVector(0, 180));
	posArray.push_back(A2DVector(30, 180)); //
	posArray.push_back(A2DVector(170, 210));
	posArray.push_back(A2DVector(320, 280));
	posArray.push_back(A2DVector(350, 280));
	posArray.push_back(A2DVector(350, 220));

	for (int a = 0; a < posArray.size(); a++)
	{
		int idx = _element_list.size();
		AnElement elem;
		elem.Triangularization(idx);
		
		// random rotation
		float radAngle = float(rand() % 628) / 100.0;
		elem.RotateXY(radAngle);

		elem.ScaleXY(initialScale);
		elem.TranslateXY(posArray[a].x, posArray[a].y);	

		elem.CalculateRestStructure();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
		elem.InitMeshOgre3D(scnMgr, pNode, "StarTube" + std::to_string(idx), "Examples/TransparentTest2");
		_element_list.push_back(elem);			
	}

	for(int a = 0; a < SystemParams::_num_layer; a++)
	{
		_c_grid_list.push_back(new CollisionGrid2D);
	}

	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			int c_grid_idx = _element_list[a]._massList[b]._layer_idx;
			A3DVector p1 = _element_list[a]._massList[b]._pos;

			
			_c_grid_list[c_grid_idx]->InsertAPoint(p1._x, p1._y, a, b); // assign mass to grid			
			_element_list[a]._massList[b]._c_grid = _c_grid_list[c_grid_idx]; // assign grid to mass
		}
	}
}

void StuffWorker::Update()
{
	// ----- for closest point calculation -----
	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].UpdateLayerBoundaries();
	}

	// ----- update collision grid -----
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

	// ----- update closest points -----
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_element_list[a]._massList[b].GetClosestPoint();
		}

	}

	// ----- grow -----
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
		_element_list[a].SolveForSprings2D();

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
		//_element_list[a].UpdateSpringDisplayOgre3D();
		_element_list[a].UpdateBoundaryDisplayOgre3D();
		_element_list[a].UpdateDebug2Ogre3D();
		//_element_list[a].UpdateClosestPtsDisplayOgre3D();
	}

}

void StuffWorker::SaveFrames()
{

	int numInterpolation = SystemParams::_interpolation_factor;

	AVideoCreator vCreator;
	vCreator.Init(numInterpolation);

	// ----- shouldn't be deleted for interpolation mode -----
	for (int l = 0; l < SystemParams::_num_layer; l++)
	{

		for (int a = 0; a < _element_list.size(); a++)
		{
			int layerOffset = l * _element_list[a]._numPointPerLayer;
			for (int b = 0; b < _element_list[a]._numBoundaryPointPerLayer; b++)
			{
				int massIdx1 = b + layerOffset;
				int massIdx2 = b + layerOffset + 1;
				if (b == _element_list[a]._numBoundaryPointPerLayer - 1)
					{ massIdx2 = layerOffset; }
				A2DVector pt1 = _element_list[a]._massList[massIdx1]._pos.GetA2DVector();
				A2DVector pt2 = _element_list[a]._massList[massIdx2]._pos.GetA2DVector();
				vCreator.DrawLine(pt1, pt2, _element_list[a]._color, l * numInterpolation);
				vCreator.DrawRedCircle(l * numInterpolation); // debug delete me

			}
		}
	}
	// ----- shouldn't be deleted for interpolation mode -----


	// WARNING very messy nested loops

	// only generate numInterpolation - 1 frames (one less)
	for (int i = 1; i < numInterpolation; i++)
	{
		float interVal = ((float)i) / ((float)numInterpolation);
		
		// one less layer
		for (int l = 0; l < SystemParams::_num_layer - 1; l++)
		{
			for (int a = 0; a < _element_list.size(); a++)
			{
				int layerOffset = l * _element_list[a]._numPointPerLayer;
				for (int b = 0; b < _element_list[a]._numBoundaryPointPerLayer; b++)
				{
					int massIdx1 = b + layerOffset;
					int massIdx2 = b + layerOffset + 1;

					if (b == _element_list[a]._numBoundaryPointPerLayer - 1)
					{
						massIdx2 = layerOffset;
					}

					int massIdx1_next = massIdx1 + _element_list[a]._numPointPerLayer; // next
					int massIdx2_next = massIdx2 + _element_list[a]._numPointPerLayer; // next

					A2DVector pt1 = _element_list[a]._massList[massIdx1]._pos.GetA2DVector();
					A2DVector pt2 = _element_list[a]._massList[massIdx2]._pos.GetA2DVector();

					A2DVector pt1_next = _element_list[a]._massList[massIdx1_next]._pos.GetA2DVector();
					A2DVector pt2_next = _element_list[a]._massList[massIdx2_next]._pos.GetA2DVector();
					

					A2DVector dir1 = pt1.DirectionTo(pt1_next);
					A2DVector dir2 = pt2.DirectionTo(pt2_next);

					float d1 = dir1.Length() * interVal;
					float d2 = dir2.Length() * interVal;

					dir1 = dir1.Norm();
					dir2 = dir2.Norm();

					A2DVector pt1_mid = pt1 + (dir1 * d1);
					A2DVector pt2_mid = pt2 + (dir2 * d2);
					
					//A2DVector pt1_mid = (pt1 + pt1_next) / 2.0;
					//A2DVector pt2_mid = (pt2 + pt2_next) / 2.0;

					int frameIdx = l * numInterpolation + i;

					vCreator.DrawLine(pt1_mid, pt2_mid, _element_list[a]._color, frameIdx);
				}
			}
		}
	}

	std::stringstream ss;
	ss << SystemParams::_save_folder << "PNG\\";
	vCreator.Save(ss.str());
}

//void StuffWorker::EnableInterpolationMode()
//{
//	// ----- variables -----
//	StuffWorker::_interpolation_mode  = true;
//	StuffWorker::_interpolation_iter  = 0;
//	StuffWorker::_interpolation_value = 0;
//
//	// -----  -----
//
//	// ----- Enable ? -----
//	for (int a = 0; a < _element_list.size(); a++)
//	{
//		_element_list[a].EnableInterpolationMode();
//	}
//	
//}
//
//void StuffWorker::DisableInterpolationMode()
//{
//	StuffWorker::_interpolation_mode  = false;
//	StuffWorker::_interpolation_iter  = 0;
//	StuffWorker::_interpolation_value = 0;
//
//	for (int a = 0; a < _element_list.size(); a++)
//	{
//		_element_list[a].DisableInterpolationMode();
//	}
//}
