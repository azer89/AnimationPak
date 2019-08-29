
#include "StuffWorker.h"
#include "OpenCVWrapper.h"
//#include "TetWrapper.h"
#include "AVideoCreator.h"
#include "ContainerWorker.h"
#include "PathIO.h"
#include "UtilityFunctions.h"
#include "PoissonGenerator.h"

#include "dirent.h" // external

// static variables
std::vector<AnElement>  StuffWorker::_element_list = std::vector<AnElement>();
CollisionGrid3D* StuffWorker::_c_grid_3d = new CollisionGrid3D;

// static variables for interpolation (need to delete these)
bool  StuffWorker::_interp_mode = false;
int   StuffWorker::_interp_iter = 0;
std::vector<CollisionGrid2D*>  StuffWorker::_interp_c_grid_list = std::vector< CollisionGrid2D * >();

StuffWorker::StuffWorker() : _containerWorker(0), _is_paused(false)
{
	_containerWorker = new ContainerWorker;
	_containerWorker->LoadContainer();

	_video_creator.Init(SystemParams::_interpolation_factor);
}

StuffWorker::~StuffWorker()
{
	if (_containerWorker) { delete _containerWorker; }

	_element_list.clear();

	if (_c_grid_3d) { delete _c_grid_3d; }
}

void StuffWorker::InitElements(Ogre::SceneManager* scnMgr)
{
	// element files
	PathIO pathIO;	
	std::vector<std::string> some_files = pathIO.LoadFiles(SystemParams::_element_folder); ////
	std::vector<std::vector<std::vector<A2DVector>>> art_paths;
	for (unsigned int a = 0; a < some_files.size(); a++)
	{
		// is path valid?
		if (some_files[a] == "." || some_files[a] == "..") { continue; }
		if (!UtilityFunctions::HasEnding(some_files[a], ".path")) { continue; }

		art_paths.push_back(pathIO.LoadElement(SystemParams::_element_folder + some_files[a]));
	}

	int elem_iter = 0;
	int elem_sz = art_paths.size();
	float initialScale = SystemParams::_element_initial_scale; // 0.05

	// docking
	//A2DVector startPt(80, 80);
	//A2DVector endPt(345, 345);
	/*A2DVector startPt(420, 80);
	A2DVector endPt(165, 345);
	{
		int idx = _element_list.size();
		AnElement elem;
		elem.Triangularization(art_paths[elem_iter++ % elem_sz], idx);
		elem.ComputeBary();
		elem.ScaleXY(initialScale);
		
		
		elem.TranslateXY(startPt.x, startPt.y);
		elem.DockEnds(startPt, endPt);
		
		elem.CalculateRestStructure();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
		elem.InitMeshOgre3D(scnMgr, pNode, "StarTube" + std::to_string(idx), "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}*/

	for (int a = 0; a < _containerWorker->_randomPositions.size(); a++)
	{
		//if (UtilityFunctions::DistanceToFiniteLine(startPt, endPt, _containerWorker->_randomPositions[a]) < 10) { continue; }

		int idx = _element_list.size();
		AnElement elem;
		elem.Triangularization(art_paths[elem_iter++ % elem_sz], idx);
		elem.ComputeBary();
		// random rotation
		float radAngle = float(rand() % 628) / 100.0;
		elem.RotateXY(radAngle);

		elem.ScaleXY(initialScale);
		elem.TranslateXY(_containerWorker->_randomPositions[a].x, _containerWorker->_randomPositions[a].y);

		elem.CalculateRestStructure();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
		elem.InitMeshOgre3D(scnMgr, pNode, "StarTube" + std::to_string(idx), "Examples/TransparentTest2");
		_element_list.push_back(elem);		

		//if (_element_list.size() == SystemParams::_num_element_pos_limit) { break; }

		//std::mt19937 g(SystemParams::_seed);
		//std::shuffle(_containerWorker->_randomPositions.begin(), _containerWorker->_randomPositions.end(), g);
	}
	
	// ----- Collision grid 3D -----
	StuffWorker::_c_grid_3d->Init();
	StuffWorker::_c_grid_3d->InitOgre3D(scnMgr);
	// ---------- Assign to collision grid 3D ----------
	for (unsigned int a = 0; a < _element_list.size(); a++)
	{
		// time triangle
		for (unsigned int b = 0; b < _element_list[a]._surfaceTriangles.size(); b++)
		{
			AnIdxTriangle tri = _element_list[a]._surfaceTriangles[b];
			A3DVector p1      = _element_list[a]._massList[tri.idx0]._pos;
			A3DVector p2      = _element_list[a]._massList[tri.idx1]._pos;
			A3DVector p3      = _element_list[a]._massList[tri.idx2]._pos;

			_c_grid_3d->InsertAPoint( (p1._x + p2._x + p3._x) * 0.333,
				                      (p1._y + p2._y + p3._y) * 0.333,
				                      (p1._z + p2._z + p3._z) * 0.333,
				                      a,  // which element
				                      b); // which triangle		
		}

		// assign
		/*for (unsigned int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_element_list[a]._massList[b]._c_grid_3d = _c_grid_3d; // assign grid to mass
		}*/
	}

	// ---------- Calculate num vertex ----------
	_num_vertex = 0;
	for (unsigned int a = 0; a < _element_list.size(); a++)
	{
		_num_vertex += _element_list[a]._massList.size();
	}

	// ----- Interpolation collision grid -----
	// INTERP WONT WORK BECAUSE OF THIS
	//for (int a = 0; a < SystemParams::_interpolation_factor; a++)
	//	{ _interp_c_grid_list.push_back(new CollisionGrid2D); }

	// ----- Assign to interpolation collision grid -----
	/*for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._interp_massList.size(); b++)
		{
			int c_grid_idx = _element_list[a]._interp_massList[b]._layer_idx;
			A3DVector p1 = _element_list[a]._interp_massList[b]._pos;

			_interp_c_grid_list[c_grid_idx]->InsertAPoint(p1._x, p1._y, a, b); // assign mass to grid			
			_element_list[a]._interp_massList[b]._c_grid = _interp_c_grid_list[c_grid_idx]; // assign grid to mass
		}
	}*/
	// INTERP WONT WORK BECAUSE OF THIS

	// debug delete me
	/*std::vector<A3DVector> tri1;
	tri1.push_back(A3DVector(300, 0, -100));
	tri1.push_back(A3DVector(400, 0, -400));
	tri1.push_back(A3DVector(0, 0, -10));
	
	
	_triangles.push_back(tri1);

	// material
	Ogre::MaterialPtr line_material = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("TriDebugLines");
	line_material->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(1, 0, 0, 1));
	_debug_lines_tri = new DynamicLines(line_material, Ogre::RenderOperation::OT_LINE_LIST);
	for (int a = 0; a < _triangles.size(); a++)
	{
		A3DVector pt1 = _triangles[a][0];
		A3DVector pt2 = _triangles[a][1];
		A3DVector pt3 = _triangles[a][2];

		_debug_lines_tri->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
		_debug_lines_tri->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));

		_debug_lines_tri->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
		_debug_lines_tri->addPoint(Ogre::Vector3(pt3._x, pt3._y, pt3._z));

		_debug_lines_tri->addPoint(Ogre::Vector3(pt3._x, pt3._y, pt3._z));
		_debug_lines_tri->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
	}


	for (int a = 0; a < 100; a++)
	{
		A3DVector randPt(rand()% 500, rand() % 500, -(rand() % 500));
		A3DVector closestPt = UtilityFunctions::ClosestPointOnTriangle2(randPt, _triangles[0][0], _triangles[0][1], _triangles[0][2]);

		_debug_lines_tri->addPoint(Ogre::Vector3(randPt._x, randPt._y, randPt._z));
		_debug_lines_tri->addPoint(Ogre::Vector3(closestPt._x, closestPt._y, closestPt._z));

	}

	_debug_lines_tri->update();
	_debugNode_tri = scnMgr->getRootSceneNode()->createChildSceneNode("debug_lines_tri_debug");
	_debugNode_tri->attachObject(_debug_lines_tri);*/

}

// INTERPOLATION
void StuffWorker::Interp_Update()
{
	// ----- for closest point calculation -----
	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].Interp_UpdateLayerBoundaries();
	}

	// ----- update collision grid -----
	std::vector<int> iters; // TODO can be better
	for (int a = 0; a < _interp_c_grid_list.size(); a++) 
		{ iters.push_back(0); }

	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._interp_massList.size(); b++)
		{
			int c_grid_idx = _element_list[a]._interp_massList[b]._layer_idx;
			int layer_iter = iters[c_grid_idx];  // why is this called layer_iter?
			A3DVector p1 = _element_list[a]._interp_massList[b]._pos;

			// update pt
			_interp_c_grid_list[c_grid_idx]->_objects[layer_iter]->_x = p1._x;
			_interp_c_grid_list[c_grid_idx]->_objects[layer_iter]->_y = p1._y;

			iters[c_grid_idx]++; // increment
		}
	}
	for (int a = 0; a < _interp_c_grid_list.size(); a++)
	{
		_interp_c_grid_list[a]->MovePoints();
		_interp_c_grid_list[a]->PrecomputeGraphIndices();
	}

	// ----- update closest points -----
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._interp_massList.size(); b++)
		{
			_element_list[a]._interp_massList[b].Interp_GetClosestPoint();
		}

	}

	// move to another layer?
	if (!Interp_HasOverlap())
	{
		Interp_SaveFrames();

		StuffWorker::_interp_iter++;

		if (StuffWorker::_interp_iter == SystemParams::_num_layer - 1)
		{
			std::stringstream ss;
			ss << SystemParams::_save_folder << "PNG\\";
			_video_creator.Save(ss.str());

			DisableInterpolationMode();
		}
		else
		{
			// ----- interpolation -----
			for (int a = 0; a < _element_list.size(); a++)
			{
				_element_list[a].UpdateInterpMasses();
			}

			// ----- Enable ? -----
			for (int a = 0; a < _element_list.size(); a++)
			{
				_element_list[a].Interp_ResetSpringRestLengths();
			}
		}
	}

}

void StuffWorker::Update()
{

	if (_is_paused) { return; }

	// ----- for closest point calculation -----
	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].UpdateLayerBoundaries();
		
	}

	// ----- update triangles -----
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._surfaceTriangles.size(); b++)
		{
			AnIdxTriangle tri = _element_list[a]._surfaceTriangles[b];
			A3DVector p1 = _element_list[a]._massList[tri.idx0]._pos;
			A3DVector p2 = _element_list[a]._massList[tri.idx1]._pos;
			A3DVector p3 = _element_list[a]._massList[tri.idx2]._pos;
			A3DVector midPt((p1._x + p2._x + p3._x) * 0.33333333333,
				            (p1._y + p2._y + p3._y) * 0.33333333333,
				            (p1._z + p2._z + p3._z) * 0.33333333333);

			_element_list[a]._surfaceTriangles[b]._temp_1_3d = p1;
			_element_list[a]._surfaceTriangles[b]._temp_2_3d = p2;
			_element_list[a]._surfaceTriangles[b]._temp_3_3d = p3;
			_element_list[a]._surfaceTriangles[b]._temp_center_3d = midPt;
			
		}
	}

	float iter = 0;
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._surfaceTriangles.size(); b++)
		{
			_c_grid_3d->SetPoint(iter++, _element_list[a]._surfaceTriangles[b]._temp_center_3d);
		}
	}	
	_c_grid_3d->MovePoints();
	_c_grid_3d->PrecomputeData();
	
	// ----- update closest points -----
	// TODO
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_element_list[a]._massList[b].GetClosestPoint4();
			//_element_list[a].UpdatePerLayerInsideFlags();
		}

	}

	// ----- grow -----
	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].Grow(SystemParams::_growth_scale_iter, SystemParams::_dt);
	}


}

void StuffWorker::Interp_Reset()
{
	// update closest points
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._interp_massList.size(); b++)
		{
			_element_list[a]._interp_massList[b].Init();
		}

	}
}

void StuffWorker::Reset()
{
	if (_is_paused) { return; }

	// update closest points
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_element_list[a]._massList[b].Init();
		}

	}
}

void StuffWorker::Interp_Solve()
{
	/*for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].Interp_SolveForSprings2D();

		for (int b = 0; b < _element_list[a]._interp_massList.size(); b++)
		{
			_element_list[a]._interp_massList[b].Solve(_containerWorker->_2d_container);
		}
	}*/
}

void StuffWorker::Solve()
{
	if (_is_paused) { return; }

	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].SolveForSprings3D();

		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_element_list[a]._massList[b].Solve(_containerWorker->_2d_container, _element_list[a]);
		}
	}
}

void StuffWorker::Interp_Simulate()
{
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._interp_massList.size(); b++)
		{
			_element_list[a]._interp_massList[b].Interp_Simulate(SystemParams::_dt);
		}
	}
}

bool StuffWorker::Interp_HasOverlap()
{
	for (int a = 0; a < _element_list.size(); a++)
	{
		if (_element_list[a].Interp_HasOverlap())
			return true;
	}
	return false;
}

void StuffWorker::Simulate()
{
	if (_is_paused) { return; }

	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_element_list[a]._massList[b].Simulate(SystemParams::_dt);
		}
	}
}

/*void StuffWorker::Interp_ImposeConstraints()
{
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._interp_massList.size(); b++)
		{
			_element_list[a]._interp_massList[b].ImposeConstraints();
		}
	}
}*/

void StuffWorker::ImposeConstraints()
{
	if (_is_paused) { return; }

	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].UpdateZConstraint();
	}

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
		_element_list[a].UpdateDockLinesOgre3D();
		_element_list[a].UpdateSurfaceTriangleOgre3D();
		_element_list[a].UpdateClosestPtsDisplayOgre3D();
		_element_list[a].UpdateOverlapOgre3D();
		_element_list[a].UpdateNegSpaceEdgeOgre3D();
		_element_list[a].UpdateMassListOgre3D();
		_element_list[a].UpdateForceOgre3D();
		_element_list[a].UpdateTimeEdgesOgre3D();
		_element_list[a].UpdateClosestSliceOgre3D();
		_element_list[a].UpdateClosestTriOgre3D();
	}

	StuffWorker::_c_grid_3d->UpdateOgre3D();
	//_element_list[0].UpdateClosestPtsDisplayOgre3D();
	//_element_list[0].UpdateClosestSliceOgre3D();

	_containerWorker->UpdateOgre3D();

}

void StuffWorker::Interp_SaveFrames()
{
	//int l = StuffWorker::_interpolation_iter;
	for (int a = 0; a < _element_list.size(); a++)
	{
		int layerOffset = StuffWorker::_interp_iter * _element_list[a]._numPointPerLayer;
		for (int b = 0; b < _element_list[a]._numBoundaryPointPerLayer; b++)
		{
			int massIdx1 = b + layerOffset;
			int massIdx2 = b + layerOffset + 1;
			if (b == _element_list[a]._numBoundaryPointPerLayer - 1)
			{
				massIdx2 = layerOffset;
			}
			A2DVector pt1 = _element_list[a]._massList[massIdx1]._pos.GetA2DVector();
			A2DVector pt2 = _element_list[a]._massList[massIdx2]._pos.GetA2DVector();
			int frameIdx = StuffWorker::_interp_iter * SystemParams::_interpolation_factor;
			_video_creator.DrawLine(pt1, pt2, _element_list[a]._color, frameIdx);
			_video_creator.DrawRedCircle(frameIdx); // debug delete me

		}
	}

	for (int i = 0; i < SystemParams::_interpolation_factor - 1; i++)
	{		
		for (int a = 0; a < _element_list.size(); a++)
		{
			int layerOffset = i * _element_list[a]._numPointPerLayer;
			for (int b = 0; b < _element_list[a]._numBoundaryPointPerLayer; b++)
			{
				int massIdx1 = b + layerOffset;
				int massIdx2 = b + layerOffset + 1;
				if (b == _element_list[a]._numBoundaryPointPerLayer - 1)
				{
					massIdx2 = layerOffset;
				}
				A2DVector pt1 = _element_list[a]._interp_massList[massIdx1]._pos.GetA2DVector();
				A2DVector pt2 = _element_list[a]._interp_massList[massIdx2]._pos.GetA2DVector();

				int frameIdx = (StuffWorker::_interp_iter * SystemParams::_interpolation_factor) + (i + 1);
				_video_creator.DrawLine(pt1, pt2, _element_list[a]._color, frameIdx);

			}
		}
	}
}

void StuffWorker::SaveFrames4()
{
	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].CalculateLayerTriangles_Drawing();
	}

	AVideoCreator vCreator;
	vCreator.Init(SystemParams::_num_png_frame);

	//for (int l = 0; l < SystemParams::_num_png_frame; l++)
	//{
	
	for (int i = 0; i < _element_list.size(); i++)
	{
		std::cout << "elem=" << i << "\n";
		MyColor col = _element_list[i]._color;
		std::vector<std::vector<std::vector<A2DVector>>> per_layer_triangle_drawing = _element_list[i]._per_layer_triangle_drawing;
		for (int l = 0; l < per_layer_triangle_drawing.size(); l++)
		{
			std::vector<std::vector<A2DVector>> triangles_in_a_layer = per_layer_triangle_drawing[l];
			std::vector<std::vector<A2DVector>> arts = _element_list[i].GetBilinearInterpolatedArt(triangles_in_a_layer);
			vCreator.DrawFilledArt(arts, col, l);
			/*for (int a = 0; a < triangles_in_a_layer.size(); a++)
			{
				// iterate triangle
				A2DVector pt1 = triangles_in_a_layer[a][0];
				A2DVector pt2 = triangles_in_a_layer[a][1];
				A2DVector pt3 = triangles_in_a_layer[a][2];

				vCreator.DrawLine(pt1, pt2, col, l);
				vCreator.DrawLine(pt2, pt3, col, l);
				vCreator.DrawLine(pt3, pt1, col, l);
			}*/
		}
	}

	std::stringstream ss;
	ss << SystemParams::_save_folder << "PNG\\";
	vCreator.Save(ss.str());
}

void StuffWorker::SaveFrames3()
{
	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].CalculateLayerTriangles_Drawing();
	}
	
	AVideoCreator vCreator;
	vCreator.Init(SystemParams::_num_png_frame);

	//for (int l = 0; l < SystemParams::_num_png_frame; l++)
	//{
	//std::cout << l << "\n";
	for (int i = 0; i < _element_list.size(); i++)
	{
		MyColor col = _element_list[i]._color;
		std::vector<std::vector<std::vector<A2DVector>>> per_layer_triangle_drawing = _element_list[i]._per_layer_triangle_drawing;
		for (int l = 0; l < per_layer_triangle_drawing.size(); l++)
		{
			std::vector<std::vector<A2DVector>> triangles_in_a_layer = per_layer_triangle_drawing[l];
			for (int a = 0; a < triangles_in_a_layer.size(); a++)
			{
				// iterate triangle
				A2DVector pt1 = triangles_in_a_layer[a][0];
				A2DVector pt2 = triangles_in_a_layer[a][1];
				A2DVector pt3 = triangles_in_a_layer[a][2];

				vCreator.DrawLine(pt1, pt2, col, l);
				vCreator.DrawLine(pt2, pt3, col, l);
				vCreator.DrawLine(pt3, pt1, col, l);
			}
		}
	}

	std::stringstream ss;
	ss << SystemParams::_save_folder << "PNG\\";
	vCreator.Save(ss.str());
}

void StuffWorker::SaveFrames2()
{
	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].CalculateLayerBoundaries_Drawing();
	}

	AVideoCreator vCreator;
	vCreator.Init(SystemParams::_num_png_frame);

	for (int l = 0; l < SystemParams::_num_png_frame; l++)
	{
		std::cout << l << "\n";
		for (int a = 0; a < _element_list.size(); a++)
		{
			for (int b = 0; b < _element_list[a]._numBoundaryPointPerLayer; b++)
			{
				int massIdx1 = b;
				int massIdx2 = b + 1;

				if (massIdx2 == _element_list[a]._numBoundaryPointPerLayer)
				{
					massIdx2 = 0;
				}
				A2DVector pt1 = _element_list[a]._per_layer_boundary_drawing[l][massIdx1].GetA2DVector();
				A2DVector pt2 = _element_list[a]._per_layer_boundary_drawing[l][massIdx2].GetA2DVector();
				vCreator.DrawLine(pt1, pt2, _element_list[a]._color, l);
			}
		}
	}

	std::stringstream ss;
	ss << SystemParams::_save_folder << "PNG\\";
	vCreator.Save(ss.str());
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

void StuffWorker::EnableInterpolationMode()
{
	std::cout << "enable interpolation\n";

	// ----- variables -----
	StuffWorker::_interp_mode  = true;
	StuffWorker::_interp_iter  = 0;
//	StuffWorker::_interpolation_value = 0;
//
//	// -----  -----
//
	_video_creator.ClearFrames();

	// ----- interpolation -----
	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].UpdateInterpMasses();
	}

	// ----- Enable ? -----
	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].Interp_ResetSpringRestLengths();
	}
//	
}

void StuffWorker::DisableInterpolationMode()
{
	std::cout << "disable interpolation\n";

	StuffWorker::_interp_mode  = false;
	StuffWorker::_interp_iter  = 0;
//	StuffWorker::_interpolation_value = 0;
//
//	for (int a = 0; a < _element_list.size(); a++)
//	{
//		_element_list[a].DisableInterpolationMode();
//	}
}

/*void StuffWorker::CreateRandomElementPoints(std::vector<A2DVector> ornamentBoundary,
									float img_length,
									std::vector<A2DVector>& randomPoints,
									int& boundaryPointNum)
{
	// how many points (really weird code...)
	float fVal = img_length / SystemParams::_upscaleFactor;
	fVal *= fVal;
	int numPoints = SystemParams::_sampling_density * fVal;
	float resamplingGap = std::sqrt(float(numPoints)) / float(numPoints) * img_length;

	std::vector<A2DVector> resampledBoundary;

	ornamentBoundary.push_back(ornamentBoundary[0]); // closed sampling
	float rGap = (float)(resamplingGap * SystemParams::_boundary_sampling_factor);
	UtilityFunctions::UniformResample(ornamentBoundary, resampledBoundary, rGap);
	// bug !!! nasty code
	if (resampledBoundary[resampledBoundary.size() - 1].Distance(resampledBoundary[0]) < rGap * 0.5) // r gap
	{
		resampledBoundary.pop_back();
	}

	PoissonGenerator::DefaultPRNG PRNG;
	if (SystemParams::_seed > 0)
	{
		PRNG = PoissonGenerator::DefaultPRNG(SystemParams::_seed);
	}
	const auto points = PoissonGenerator::GeneratePoissonPoints(numPoints, PRNG);

	randomPoints.insert(randomPoints.begin(), resampledBoundary.begin(), resampledBoundary.end());
	boundaryPointNum = resampledBoundary.size();

	float sc = img_length * std::sqrt(2.0f);
	float ofVal = 0.5f * (sc - img_length);
	//float ofVal = 0;
	// ---------- iterate points ----------
	for (auto i = points.begin(); i != points.end(); i++)
	{
		float x = (i->x * sc) - ofVal;
		float y = (i->y * sc) - ofVal;
		A2DVector pt(x, y);

		if (UtilityFunctions::InsidePolygon(ornamentBoundary, pt.x, pt.y))
		{
			float d = UtilityFunctions::DistanceToClosedCurve(resampledBoundary, pt);
			if (d > resamplingGap)
			{
				randomPoints.push_back(pt);
			}
			//AVector cPt = knn->GetClosestPoints(pt, 1)[0];
			//if (cPt.Distance(pt) > resamplingGap)
			//	{ randomPoints.push_back(pt); }
		}
	}
}*/
