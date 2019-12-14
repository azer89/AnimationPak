
#include "StuffWorker.h"
#include "OpenCVWrapper.h"
//#include "TetWrapper.h"
#include "AVideoCreator.h"
#include "ContainerWorker.h"
#include "PathIO.h"
#include "UtilityFunctions.h"
#include "PoissonGenerator.h"

#include "dirent.h" // external

#include <chrono> // debug delete me
#include <algorithm> 

// static variables
std::vector<AnElement>  StuffWorker::_element_list = std::vector<AnElement>();
CollisionGrid3D* StuffWorker::_c_grid_3d = new CollisionGrid3D;

// static variables for interpolation (need to delete these)
//bool  StuffWorker::_interp_mode = false;
//int   StuffWorker::_interp_iter = 0;
//std::vector<CollisionGrid2D*>  StuffWorker::_interp_c_grid_list = std::vector< CollisionGrid2D * >();

StuffWorker::StuffWorker() : _containerWorker(0), _is_paused(false), _my_thread_pool(0)
{
	//_almostall_multi_t = 0;
	//_almostall_single_t = 0;
	//_cg_multi_t = 0;
	//_cg_single_t = 0;
	_num_iteration = 0;

	_max_c_pts = 0;
	_max_c_pts_approx = 0;

	_containerWorker = new ContainerWorker;
	_containerWorker->LoadContainer();

	_my_thread_pool = new ThreadPool(SystemParams::_num_threads);
	//_video_creator.Init(SystemParams::_interpolation_factor);
}

StuffWorker::~StuffWorker()
{
	if (_containerWorker) { delete _containerWorker; }
	std::cout << "container worker destroyed\n";	
	if (_my_thread_pool) { delete _my_thread_pool; }
	std::cout << "threadpool destroyed\n";
	
	if (_c_grid_3d) { delete _c_grid_3d; }
	std::cout << "collision grid destroyed\n";

	_element_list.clear();
	std::cout << "elements destroyed\n";
}

void StuffWorker::DockElementsOnPaths(std::vector <std::vector<A3DVector>> paths,
	                           std::vector<std::vector<int>> layer_indices,
	                           std::vector<AnElement> temp_elements,
	                           Ogre::SceneManager* scnMgr)
{
	int temp_elem_sz = temp_elements.size();
	float initialScale = SystemParams::_element_initial_scale; // 0.05

	for (int a = 0; a < paths.size(); a++)
	{
		int idx = a;

		//AnElement elem = temp_elements[idx % temp_elem_sz];
		AnElement elem = temp_elements[a];
		elem.TriangularizationThatIsnt(idx);

		// for marine_life
		//elem.CreateHelix(0.5);

		float len = paths[a].size();

		// TODO: from one dockpoint to the next one, not start to finish
		A2DVector move_dir = paths[a][0].GetA2DVector().DirectionTo(paths[a][len - 1].GetA2DVector());
		float radAngle = UtilityFunctions::Angle2D(0, 1, move_dir.x, move_dir.y);
		elem.RotateXY(radAngle);

		elem.ScaleXY(initialScale);

		// TODO: more than two dock points
		A2DVector startPt = paths[a][0].GetA2DVector();
		A2DVector endPt = paths[a][len - 1].GetA2DVector();

		//elem.TranslateXY(startPt.x, startPt.y);

		elem.UpdateLayerBoundaries(); // per_layer_boundary
		elem.CalculateRestStructure(); // calculate rest, why???

		//elem.DockEnds(startPt, endPt); // docking
		elem.Docking(paths[a], layer_indices[a]);

		elem.CalculateRestStructure(); // calculate rest, why???

		// script for rotating arms
		elem.AddConnector(idx,
							0,
							SystemParams::_num_layer - 1);

		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
		elem.InitMeshOgre3D(scnMgr, pNode, "Tube_" + std::to_string(idx), "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	

	// script for marine_life (delete above connector)
	/*int last_layer_idx = SystemParams::_num_layer - 1;
	_element_list[0].AddConnector(1,               // other_elem_idx
		                          0,               // ur_layer_idx
		                          last_layer_idx); // their_layer_idx

	_element_list[1].AddConnector(0,               // other_elem_idx
		                          last_layer_idx,  // ur_layer_idx
		                          0);              // their_layer_idx

	// scripted!
	_element_list[0].AddConnector(1,              // other_elem_idx
		                          last_layer_idx, // ur_layer_idx
		                          0);             // their_layer_idx

	_element_list[1].AddConnector(0,              // other_elem_idx
		                          0,              // ur_layer_idx
		                          last_layer_idx);// their_layer_idx*/
}

void StuffWorker::ConnectTubeEnds()
{
	// why does this function exist?

}

void StuffWorker::InitSavedScenes(Ogre::SceneManager* scnMgr)
{
	// element files
	PathIO pathIO;

	std::vector<std::string> some_files = pathIO.LoadFiles(SystemParams::_animated_element_folder); ////

	for (unsigned int a = 0; a < some_files.size(); a++)
	{
		// is path valid?
		if (some_files[a] == "." || some_files[a] == "..") { continue; }
		if (!UtilityFunctions::HasEnding(some_files[a], ".path")) { continue; }

		_element_list.push_back(pathIO.LoadAnimatedElement(SystemParams::_animated_element_folder + some_files[a]));
	}

	for (int a = 0; a < _element_list.size(); a++)
	{
		//int idx = _element_list.size();

		//AnElement elem = temp_elements[1];

		_element_list[a].TriangularizationThatIsnt(a);

		//float radAngle = float(rand() % 628) / 100.0;
		//elem.RotateXY(radAngle);

		//elem.ScaleXY(initialScale);
		//elem.TranslateXY(_containerWorker->_randomPositions[a].x, _containerWorker->_randomPositions[a].y);
		//elem.TranslateXY(positions[a].x, positions[a].y);

		_element_list[a].CalculateRestStructure();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(a));
		_element_list[a].InitMeshOgre3D(scnMgr, pNode, "Tube_" + std::to_string(a), "Examples/TransparentTest2");


		// don't work...
		//_element_list[a].AddConnector(idx, 0, SystemParams::_num_layer - 1);
		

		//_element_list.push_back(elem);



		// dumb code
		//if (_element_list.size() == SystemParams::_num_element_pos_limit) { break; }
	}
}

void StuffWorker::JitterPosAndRotation(float pos_max_offset, A2DVector& pos_offset, float& rot_val)
{
	// https_//stackoverflow.com/questions/686353/random-float-number-generation
	float twice_offset = pos_max_offset * 2.0f;

	pos_offset.x = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / pos_max_offset)) - twice_offset;
	pos_offset.y = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / pos_max_offset)) - twice_offset;

	float PI = 3.14159265359;

	rot_val = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / PI));
}

void StuffWorker::InitRotatingArms(Ogre::SceneManager* scnMgr)
{

	// element files
	PathIO pathIO;

	// scene
	std::vector <std::vector<A3DVector>> paths;
	std::vector<std::vector<int>> layer_indices;
	std::vector<A2DVector> positions;
	pathIO.LoadScenes(paths, layer_indices, positions, SystemParams::_scene_file_name);

	// elements
	std::vector<std::string> some_files = pathIO.LoadFiles(SystemParams::_animated_element_folder); ////
	std::vector<AnElement> temp_elements;
	for (unsigned int a = 0; a < some_files.size(); a++)
	{
		// is path valid?
		if (some_files[a] == "." || some_files[a] == "..") { continue; }
		if (!UtilityFunctions::HasEnding(some_files[a], ".path")) { continue; }

		temp_elements.push_back(pathIO.LoadAnimatedElement(SystemParams::_animated_element_folder + some_files[a]));
	}

	//int elem_iter = 0;
	int temp_elem_sz = temp_elements.size();
	float initialScale = SystemParams::_element_initial_scale; // 0.05

	//DockElementsOnPaths(paths, layer_indices, temp_elements, scnMgr);
	std::cout << "+++++++++++++ path size = " << paths.size() << "\n";
	for (int a = 0; a < paths.size(); a++)
	{
		int idx = a;

		AnElement elem = temp_elements[a];
		//elem.SetIndex(idx);

		elem.TriangularizationThatIsnt(idx);

		//float radAngle = float(rand() % 628) / 100.0;
		//elem.RotateXY(radAngle);
		//float radAngle;
		//A2DVector offset;
		//JitterPosAndRotation(3, offset, radAngle);
		//elem.RotateXY(radAngle);

		// ROTATING ARMS
		elem._is_rotating_arms = true;

		elem.ScaleXY(initialScale);
		//elem.TranslateXY(_containerWorker->_randomPositions[a].x, _containerWorker->_randomPositions[a].y);
		//elem.TranslateXY(positions[a].x, positions[a].y);
		elem.MoveXY(paths[a][0]._x, paths[a][0]._y);

		elem.CalculateRestStructure();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
		elem.InitMeshOgre3D(scnMgr, pNode, "Tube_" + std::to_string(idx), "Examples/TransparentTest2");

		// other_elem_idx
		// int ur_layer_idx 
		// int their_layer_idx
		//elem.AddConnector(idx,
		//	0,
		//	SystemParams::_num_layer - 1);



		_element_list.push_back(elem);



		// dumb code
		//if (_element_list.size() == SystemParams::_num_element_pos_limit) { break; }
	}


	std::random_shuffle(positions.begin(), positions.end());

	// NON-DOCKED ELEMENTS
	for (int a = 0; a < positions.size(); a++)
	{
		int idx = _element_list.size();

		int num_dock_elem = 2; // CHANGE THIS
		int temp_elem_idx = (a % (temp_elem_sz - num_dock_elem)) + num_dock_elem;
		AnElement elem = temp_elements[temp_elem_idx];
		//elem.SetIndex(idx);

		elem.TriangularizationThatIsnt(idx);

		//float radAngle = float(rand() % 628) / 100.0;
		//elem.RotateXY(radAngle);
		//float radAngle;
		//A2DVector offset;
		//JitterPosAndRotation(3, offset, radAngle);
		//elem.RotateXY(radAngle);

		elem.ScaleXY(initialScale);
		//elem.TranslateXY(_containerWorker->_randomPositions[a].x, _containerWorker->_randomPositions[a].y);
		//elem.TranslateXY(positions[a].x, positions[a].y);
		elem.MoveXY(positions[a].x, positions[a].y);

		elem.CalculateRestStructure();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
		elem.InitMeshOgre3D(scnMgr, pNode, "Tube_" + std::to_string(idx), "Examples/TransparentTest2");

		// other_elem_idx
		// int ur_layer_idx 
		// int their_layer_idx
		//elem.AddConnector(idx,
		//	0,
		//	SystemParams::_num_layer - 1);



		_element_list.push_back(elem);



		// dumb code
		//if (_element_list.size() == SystemParams::_num_element_pos_limit) { break; }
	}

	std::cout << "Elements done...\n";
}

// USE THIS!!!!
void StuffWorker::InitAnimated_Elements(Ogre::SceneManager* scnMgr)
{
	// element files
	PathIO pathIO;

	// scene
	std::vector <std::vector<A3DVector>> paths;
	std::vector<std::vector<int>> layer_indices;
	std::vector<A2DVector> positions;
	pathIO.LoadScenes(paths, layer_indices, positions, SystemParams::_scene_file_name);

	// elements
	std::vector<std::string> some_files = pathIO.LoadFiles(SystemParams::_animated_element_folder); ////
	std::vector<AnElement> temp_elements;
	for (unsigned int a = 0; a < some_files.size(); a++)
	{
		// is path valid?
		if (some_files[a] == "." || some_files[a] == "..") { continue; }
		if (!UtilityFunctions::HasEnding(some_files[a], ".path")) { continue; }

		temp_elements.push_back(pathIO.LoadAnimatedElement(SystemParams::_animated_element_folder + some_files[a]));
	}

	//int elem_iter = 0;
	int temp_elem_sz = temp_elements.size();
	float initialScale = SystemParams::_element_initial_scale; // 0.05
	 
	DockElementsOnPaths(paths, layer_indices, temp_elements, scnMgr);

	

	std::random_shuffle(positions.begin(), positions.end());

	// NON-DOCKED ELEMENTS
	for (int a = 0; a < positions.size(); a++)
	{
		int idx = _element_list.size();

		int num_dock_elem = 2; // CHANGE THIS
		int temp_elem_idx = (a % (temp_elem_sz - num_dock_elem)) + num_dock_elem;
		AnElement elem = temp_elements[temp_elem_idx];
		//elem.SetIndex(idx);

		elem.TriangularizationThatIsnt(idx);

		//float radAngle = float(rand() % 628) / 100.0;
		//elem.RotateXY(radAngle);
		//float radAngle;
		//A2DVector offset;
		//JitterPosAndRotation(3, offset, radAngle);
		//elem.RotateXY(radAngle);

		elem.ScaleXY(initialScale);
		//elem.TranslateXY(_containerWorker->_randomPositions[a].x, _containerWorker->_randomPositions[a].y);
		//elem.TranslateXY(positions[a].x, positions[a].y);
		elem.MoveXY(positions[a].x, positions[a].y);

		elem.CalculateRestStructure();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
		elem.InitMeshOgre3D(scnMgr, pNode, "Tube_" + std::to_string(idx), "Examples/TransparentTest2");
		
		// other_elem_idx
		// int ur_layer_idx 
		// int their_layer_idx
		elem.AddConnector(idx, 
			              0, 
			              SystemParams::_num_layer - 1);

		
		
		_element_list.push_back(elem);



		// dumb code
		if (_element_list.size() == SystemParams::_num_element_pos_limit) { break; }
	}


	// debugging
	//DynamicLines*    _pos_lines;
	//Ogre::SceneNode* _pos_node;
	/*Ogre::MaterialPtr _pos_material = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("_pos_material_");
	_pos_material->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(1, 0, 0, 1));
	_pos_lines = new DynamicLines(_pos_material, Ogre::RenderOperation::OT_LINE_LIST);

	for (int a = 0; a < paths.size(); a++)
	{
		A3DVector pt = paths[a][0];

		float offsetVal = 2;
		_pos_lines->addPoint(Ogre::Vector3(pt._x - offsetVal, pt._y, 0));
		_pos_lines->addPoint(Ogre::Vector3(pt._x + offsetVal, pt._y, 0));
		_pos_lines->addPoint(Ogre::Vector3(pt._x, pt._y - offsetVal, 0));
		_pos_lines->addPoint(Ogre::Vector3(pt._x, pt._y + offsetVal, 0));
	}


	for (int a = 0; a < positions.size(); a++)
	{
		A3DVector pt(positions[a].x, positions[a].y, 0);

		float offsetVal = 2;
		_pos_lines->addPoint(Ogre::Vector3(pt._x - offsetVal, pt._y, 0));
		_pos_lines->addPoint(Ogre::Vector3(pt._x + offsetVal, pt._y, 0));
		_pos_lines->addPoint(Ogre::Vector3(pt._x, pt._y - offsetVal, 0));
		_pos_lines->addPoint(Ogre::Vector3(pt._x, pt._y + offsetVal, 0));
	}

	_pos_lines->update();
	_pos_node = scnMgr->getRootSceneNode()->createChildSceneNode("_ps_node_");
	_pos_node->attachObject(_pos_lines);*/

	std::cout << "Elements done...\n";
}

// DON'T USE THIS!!!!
void StuffWorker::InitElements_OneMovingElement(Ogre::SceneManager* scnMgr)
{
	// element files
	PathIO pathIO;
	std::vector<std::string> some_files = pathIO.LoadFiles(SystemParams::_static_element_folder); ////
	std::vector<std::vector<std::vector<A2DVector>>> art_paths;
	for (unsigned int a = 0; a < some_files.size(); a++)
	{
		// is path valid?
		if (some_files[a] == "." || some_files[a] == "..") { continue; }
		if (!UtilityFunctions::HasEnding(some_files[a], ".path")) { continue; }

		art_paths.push_back(pathIO.LoadElement(SystemParams::_static_element_folder + some_files[a]));
	}

	//int elem_iter = 0;
	int art_sz = art_paths.size();
	float initialScale = SystemParams::_element_initial_scale; // 0.05

	std::vector<AnElement> tempElems;
	for (int a = 0; a < art_sz; a++)
	{
		AnElement elem;
		elem.Triangularization(art_paths[a], a);
		elem.ComputeBary();

		tempElems.push_back(elem);
	}

	std::cout << "Triangulation done...\n";

	A2DVector startPt(100, 100);
	A2DVector endPt(400, 400);
	{
		int idx = 0;

		AnElement elem = tempElems[idx % art_sz];
		elem.SetIndex(idx);

		float radAngle = float(rand() % 628) / 100.0;
		elem.RotateXY(radAngle);

		elem.ScaleXY(initialScale);
		elem.TranslateXY(startPt.x, startPt.y);

		elem.UpdateLayerBoundaries(); // per_layer_boundary
		elem.CalculateRestStructure(); // calculate rest

		elem.DockEnds(startPt, endPt); // docking

		elem.CalculateRestStructure(); // calculate rest

		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
		elem.InitMeshOgre3D(scnMgr, pNode, "Tube_" + std::to_string(idx), "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	for (int a = 0; a < SystemParams::_num_element_pos_limit; a++)
	{
		int idx = _element_list.size();

		AnElement elem = tempElems[idx % art_sz];
		elem.SetIndex(idx);

		float radAngle = float(rand() % 628) / 100.0;
		elem.RotateXY(radAngle);

		elem.ScaleXY(initialScale);
		elem.TranslateXY(_containerWorker->_randomPositions[a].x, _containerWorker->_randomPositions[a].y);

		elem.CalculateRestStructure();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
		elem.InitMeshOgre3D(scnMgr, pNode, "Tube_" + std::to_string(idx), "Examples/TransparentTest2");
		_element_list.push_back(elem);

		// dumb code
		if (_element_list.size() == SystemParams::_num_element_pos_limit) { break; }
	}

	std::cout << "Elements done...\n";
}

// DON'T USE THIS!!!!
void StuffWorker::InitElements_TwoMovingElements(Ogre::SceneManager* scnMgr)
{
	// element files
	PathIO pathIO;
	std::vector<std::string> some_files = pathIO.LoadFiles(SystemParams::_static_element_folder); ////
	std::vector<std::vector<std::vector<A2DVector>>> art_paths;
	for (unsigned int a = 0; a < some_files.size(); a++)
	{
		// is path valid?
		if (some_files[a] == "." || some_files[a] == "..") { continue; }
		if (!UtilityFunctions::HasEnding(some_files[a], ".path")) { continue; }

		art_paths.push_back(pathIO.LoadElement(SystemParams::_static_element_folder + some_files[a]));
	}

	//int elem_iter = 0;
	int art_sz = art_paths.size();
	float initialScale = SystemParams::_element_initial_scale; // 0.05

	// docking	
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

	//_element_list = std::vector<AnElement>(SystemParams::_num_element_pos_limit);

	std::vector<AnElement> tempElems;
	for (int a = 0; a < art_sz; a++)
	{
		AnElement elem;
		elem.Triangularization(art_paths[a], a);
		elem.ComputeBary();

		tempElems.push_back(elem);
	}

	std::cout << "Triangulation done...\n";

	A2DVector startPt(90, 90);
	A2DVector endPt(410, 410);
	{
		int idx = 0;

		AnElement elem = tempElems[idx % art_sz];
		elem.SetIndex(idx);

		float radAngle = float(rand() % 628) / 100.0;
		elem.RotateXY(radAngle);

		elem.ScaleXY(initialScale);
		elem.TranslateXY(startPt.x, startPt.y);
		elem.DockEnds(startPt, endPt);

		elem.CalculateRestStructure();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
		elem.InitMeshOgre3D(scnMgr, pNode, "Tube_" + std::to_string(idx), "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	A2DVector startPt2(400, 90);
	A2DVector endPt2(100, 410);
	{
		int idx = 1;

		AnElement elem = tempElems[idx % art_sz];
		elem.SetIndex(idx);

		float radAngle = float(rand() % 628) / 100.0;
		elem.RotateXY(radAngle);

		elem.ScaleXY(initialScale);
		elem.TranslateXY(startPt2.x, startPt2.y);
		elem.DockEnds(startPt2, endPt2);

		elem.CalculateRestStructure();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
		elem.InitMeshOgre3D(scnMgr, pNode, "Tube_" + std::to_string(idx), "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}

	for (int a = 0; a < SystemParams::_num_element_pos_limit; a++)
	{
		int idx = _element_list.size();

		AnElement elem = tempElems[idx % art_sz];
		elem.SetIndex(idx);

		float radAngle = float(rand() % 628) / 100.0;
		elem.RotateXY(radAngle);

		elem.ScaleXY(initialScale);
		elem.TranslateXY(_containerWorker->_randomPositions[a].x, _containerWorker->_randomPositions[a].y);

		elem.CalculateRestStructure();
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
		elem.InitMeshOgre3D(scnMgr, pNode, "Tube_" + std::to_string(idx), "Examples/TransparentTest2");
		_element_list.push_back(elem);

		// dumb code
		if (_element_list.size() == SystemParams::_num_element_pos_limit) { break; }
	}

	std::cout << "Elements done...\n";
}

// THIS WILL CALL ANOTHER FUNCTION
void StuffWorker::InitElementsAndCGrid(Ogre::SceneManager* scnMgr)
{
	// Your scene here!
	//InitElements_TwoMovingElements(scnMgr);
	//InitElements_OneMovingElement(scnMgr);
	//InitAnimated_Elements(scnMgr);
	InitRotatingArms(scnMgr);
	//InitSavedScenes(scnMgr);  <-- only for reloading finished simulation

	// ----- Collision grid 3D -----
	StuffWorker::_c_grid_3d->Init();
	StuffWorker::_c_grid_3d->InitOgre3D(scnMgr);
	// ---------- Assign to collision grid 3D ----------
	for (unsigned int a = 0; a < _element_list.size(); a++)
	{
		// time triangle
		for (unsigned int b = 0; b < _element_list[a]._surfaceTriangles.size(); b++)
		{
			_element_list[a].InitSurfaceTriangleMidPts();

			AnIdxTriangle tri = _element_list[a]._surfaceTriangles[b];

			_c_grid_3d->InsertAPoint(tri._temp_center_3d._x,
				tri._temp_center_3d._y,
				tri._temp_center_3d._z,
				a,  // which element
				b); // which triangle
		}
	}
	std::cout << "Collision grid done...\n";

	// ---------- Calculate num vertex ----------
	_num_vertex = 0;
	for (unsigned int a = 0; a < _element_list.size(); a++)
	{
		_num_vertex += _element_list[a]._massList.size();
	}
}

void StuffWorker::SaveScene()
{
	PathIO pIO;
	for (int a = 0; a < _element_list.size(); a++)
	{
		pIO.SaveAnimatedElement(_element_list[a], SystemParams::_save_folder + _element_list[a]._name + ".path");
	}
}

void StuffWorker::SaveStatistics()
{
	std::stringstream ss;

	// time (_cg_move_points, cg_multi_t, _almostall_multi_t)
	ss << "_cg_move_points (seconds)    = " << _cg_move_points.GetTotal() / 1000000 << "\n";
	ss << "_cg_multi_t (seconds)        = " << _cg_multi_t.GetTotal() / 1000000 << "\n";
	ss << "_almostall_multi_t (seconds) = " << _almostall_multi_t.GetTotal() / 1000000 << "\n";
	
	// total time _cg_move_points + cg_multi_t + _almostall_multi_t
	ss << "total (microsec)             = " << (_cg_move_points.GetTotal() + _cg_multi_t.GetTotal() + _almostall_multi_t.GetTotal()) / 1000000 << "\n";

	// avg skin offset ??? NOPE save the scene!

	// https_//stackoverflow.com/questions/1406029/how-to-calculate-the-volume-of-a-3d-mesh-object-the-surface-of-which-is-made-up
	// fill ratio (area of triangles, approx) ??? NOPE save the scene!

	// num elements
	ss << "num elements = " << _element_list.size() << "\n";

	// num layer springs
	int num_l_springs = 0;
	for (int a = 0; a < _element_list.size(); a++)
	{
		num_l_springs += _element_list[a]._layer_springs.size();
	}
	ss << "num layer springs = " << num_l_springs << "\n";

	// num aux springs
	int num_aux_springs = 0;
	for (int a = 0; a < _element_list.size(); a++)
	{
		num_aux_springs += _element_list[a]._auxiliary_springs.size();
	}
	ss << "num auxiliary springs = " << num_aux_springs << "\n";

	// num time springs
	int num_time_springs = 0;
	for (int a = 0; a < _element_list.size(); a++)
	{
		num_time_springs += _element_list[a]._time_springs.size();
	}
	ss << "num time springs = " << num_time_springs << "\n";

	// num neg space springs
	int num_neg_space_springs = 0;
	for (int a = 0; a < _element_list.size(); a++)
	{
		num_neg_space_springs += _element_list[a]._neg_space_springs.size();
	}
	ss << "num neg space springs = " << num_neg_space_springs << "\n";

	// num all springs
	ss << "num all springs = " << num_l_springs + num_aux_springs + num_time_springs + num_neg_space_springs << "\n";

	//  num vertices
	int num_v = 0;
	for (int a = 0; a < _element_list.size(); a++)
	{
		num_v += _element_list[a]._massList.size();
	}
	ss << "num vertices = " << num_v << "\n";

	// num triangles
	int num_t = 0;
	for (int a = 0; a < _element_list.size(); a++)
	{
		num_t += _element_list[a]._triangles.size();
	}
	ss << "num triangles = " << num_t << "\n";

	// seeds
	ss << "seeds = " << SystemParams::_seed << "\n";


	PathIO pIO;
	pIO.SaveText(ss.str(), SystemParams::_save_folder + "run_info.txt");
}

int StuffWorker::StillGrowing()
{
	int ctr = 0;
	for (int a = 0; a < _element_list.size(); a++)
	{
		ctr += _element_list[a].StillGrowing();
	}
	return ctr;
}


void StuffWorker::Update()
{
	if (_is_paused) { return; }

	// for statistics!
	_num_iteration++;

	// Collision grid
	auto start0 = std::chrono::steady_clock::now(); // timing
	float iter = 0;
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._surfaceTriangles.size(); b++)
		{
			_c_grid_3d->SetPoint(iter++, _element_list[a]._surfaceTriangles[b]._temp_center_3d);
		}
	}	
	_c_grid_3d->MovePoints();
	auto elapsed0 = std::chrono::steady_clock::now() - start0; // timing
	_cg_move_points.AddTime(std::chrono::duration_cast<std::chrono::microseconds>(elapsed0).count());
	
	if (SystemParams::_multithread_test)
	{
		// Collision grid
		auto start1_c = std::chrono::steady_clock::now(); // timing
		_c_grid_3d->PrecomputeData();
		auto elapsed1_c = std::chrono::steady_clock::now() - start1_c; // timing
		_cg_single_t.AddTime(std::chrono::duration_cast<std::chrono::microseconds>(elapsed1_c).count()); // timing
	}
	
	// ~~~~~ T ~~~~~
	auto start1 = std::chrono::steady_clock::now(); // timing
	//_c_grid_3d->PrecomputeData_Prepare_Threads();
	CollisionGrid_PrepareThreadPool();
	auto elapsed1 = std::chrono::steady_clock::now() - start1; // timing
	_cg_multi_t.AddTime( std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count() ); // timing
	// ~~~~~ T ~~~~~

	// statistics of c_pt and c_pt_approx
	_max_c_pts = 0;
	_max_c_pts_approx = 0;
	/*for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_max_c_pts = std::max(_max_c_pts, _element_list[a]._massList[b]._c_pts_fill_size);
			_max_c_pts_approx = std::max(_max_c_pts_approx, _element_list[a]._massList[b]._c_pts_approx_fill_size);
		}
	}*/
	for (int a = 0; a < _c_grid_3d->_squares.size(); a++)
	{
		_max_c_pts = std::max(_max_c_pts, _c_grid_3d->_squares[a]->_c_pt_fill_size);
		_max_c_pts_approx = std::max(_max_c_pts_approx, _c_grid_3d->_squares[a]->_c_pt_approx_fill_size);
	}

	//// ----- update closest points -----
	//if (SystemParams::_multithread_test)
	//{
	//	auto start2_c = std::chrono::steady_clock::now(); // timing
	//	for (int a = 0; a < _element_list.size(); a++)
	//	{
	//		for (int b = 0; b < _element_list[a]._massList.size(); b++)
	//		{
	//			_element_list[a]._massList[b].GetClosestPoint4();
	//		}
	//	}
	//	auto elapsed2_c = std::chrono::steady_clock::now() - start2_c; // timing
	//	_c_pt_cpu_t = std::chrono::duration_cast<std::chrono::microseconds>(elapsed2_c).count(); // timing
	//}

	//auto start2 = std::chrono::steady_clock::now(); // timing
	//GetClosestPt_Prepare_Threads();
	//auto elapsed2 = std::chrono::steady_clock::now() - start2; // timing
	//_c_pt_thread_t = std::chrono::duration_cast<std::chrono::microseconds>(elapsed2).count(); // timing

	

	//// ----- for closest point calculation -----
	//for (int a = 0; a < _element_list.size(); a++)
	//{
	//	_element_list[a].UpdateLayerBoundaries();

	//}

	//// ----- update triangles -----
	//for (int a = 0; a < _element_list.size(); a++)
	//{
	//	for (int b = 0; b < _element_list[a]._surfaceTriangles.size(); b++)
	//	{
	//		AnIdxTriangle tri = _element_list[a]._surfaceTriangles[b];
	//		A3DVector p1 = _element_list[a]._massList[tri.idx0]._pos;
	//		A3DVector p2 = _element_list[a]._massList[tri.idx1]._pos;
	//		A3DVector p3 = _element_list[a]._massList[tri.idx2]._pos;
	//		A3DVector midPt((p1._x + p2._x + p3._x) * 0.33333333333,
	//						(p1._y + p2._y + p3._y) * 0.33333333333,
	//						(p1._z + p2._z + p3._z) * 0.33333333333);

	//		_element_list[a]._surfaceTriangles[b]._temp_1_3d = p1;
	//		_element_list[a]._surfaceTriangles[b]._temp_2_3d = p2;
	//		_element_list[a]._surfaceTriangles[b]._temp_3_3d = p3;
	//		_element_list[a]._surfaceTriangles[b]._temp_center_3d = midPt;

	//	}
	//}


	//// ----- grow -----
	//for (int a = 0; a < _element_list.size(); a++)
	//{
	//	//UpdatePerLayerInsideFlags()
	//	_element_list[a].UpdatePerLayerInsideFlags();
	//	_element_list[a].Grow(SystemParams::_growth_scale_iter, SystemParams::_dt);
	//}

}

void  StuffWorker::AlmostAllUrShit()
{
	if (SystemParams::_multithread_test)
	{
		auto start1 = std::chrono::steady_clock::now(); // timing
		AlmostAllUrShit_SingleThread();
		auto elapsed1 = std::chrono::steady_clock::now() - start1; // timing
		_almostall_single_t.AddTime(std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count()); // timing
	}
	auto start2 = std::chrono::steady_clock::now(); // timing
	AlmostAllUrShit_PrepareThreadPool();
	auto elapsed2 = std::chrono::steady_clock::now() - start2; // timing
	_almostall_multi_t.AddTime(std::chrono::duration_cast<std::chrono::microseconds>(elapsed2).count()); // timing
}

void  StuffWorker::AlmostAllUrShit_SingleThread()
{
	if (_is_paused) { return; }

	// ----- UPDATE ----- 
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_element_list[a]._massList[b].GetClosestPoint4();
		}
	}

	// ----- stuff -----
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


	// ----- grow -----
	for (int a = 0; a < _element_list.size(); a++)
	{
		//UpdatePerLayerInsideFlags()
		_element_list[a].UpdatePerLayerInsideFlags();
		_element_list[a].Grow(SystemParams::_growth_scale_iter, SystemParams::_dt);
	}

	
	// ----- RESET ----- 
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_element_list[a]._massList[b].Init();
		}

	}

	// -----  SOLVE ----- 
	for (int a = 0; a < _element_list.size(); a++)
	{
		_element_list[a].SolveForSprings3D();
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_element_list[a]._massList[b].Solve(_containerWorker->_2d_container, _element_list[a]);
		}
	}

	// ----- SIMULATE ----- 
	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_element_list[a]._massList[b].Simulate(SystemParams::_dt);
		}
	}

	// -----  IMPOSE CONSTRAINT ----- 
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

void StuffWorker::AlmostAllUrShit_PrepareThreadPool()
{
	int len = _element_list.size();
	int num_threads = SystemParams::_num_threads;
	int thread_stride = (len + num_threads - 1) / num_threads;

	for (int a = 0; a < num_threads; a++)
	{
		int startIdx = a * thread_stride;
		int endIdx = startIdx + thread_stride;
		_my_thread_pool->submit(&StuffWorker::AlmostAllUrShit_ThreadTask, this, startIdx, endIdx);
	}

	_my_thread_pool->waitFinished();
}

void StuffWorker::AlmostAllUrShit_ThreadTask(int startIdx, int endIdx)
{
	for (unsigned int iter = startIdx; iter < endIdx; iter++)
	{
		// make sure...
		if (iter >= _element_list.size()) { break; }

		// UPDATE
		for (int b = 0; b < _element_list[iter]._massList.size(); b++)
		{
			_element_list[iter]._massList[b].GetClosestPoint5();
		}

		// stuff
		_element_list[iter].UpdateLayerBoundaries();

		// ----- update triangles -----
		// IS THIS THE PROPER PLACE???
		for (int b = 0; b < _element_list[iter]._surfaceTriangles.size(); b++)
		{
			AnIdxTriangle tri = _element_list[iter]._surfaceTriangles[b];
			A3DVector p1 = _element_list[iter]._massList[tri.idx0]._pos;
			A3DVector p2 = _element_list[iter]._massList[tri.idx1]._pos;
			A3DVector p3 = _element_list[iter]._massList[tri.idx2]._pos;
			A3DVector midPt((p1._x + p2._x + p3._x) * 0.33333333333,
							(p1._y + p2._y + p3._y) * 0.33333333333,
							(p1._z + p2._z + p3._z) * 0.33333333333);

			_element_list[iter]._surfaceTriangles[b]._temp_1_3d = p1;
			_element_list[iter]._surfaceTriangles[b]._temp_2_3d = p2;
			_element_list[iter]._surfaceTriangles[b]._temp_3_3d = p3;
			_element_list[iter]._surfaceTriangles[b]._temp_center_3d = midPt;

		}

		// ----- grow -----
		_element_list[iter].UpdatePerLayerInsideFlags();
		_element_list[iter].Grow(SystemParams::_growth_scale_iter, SystemParams::_dt);

		// RESET
		for (int b = 0; b < _element_list[iter]._massList.size(); b++)
		{
			_element_list[iter]._massList[b].Init();
		}

		// SOLVE
		_element_list[iter].SolveForSprings3D();
		for (int b = 0; b < _element_list[iter]._massList.size(); b++)
		{
			_element_list[iter]._massList[b].Solve(_containerWorker->_2d_container, _element_list[iter]);
		}

		// SIMULATE
		for (int b = 0; b < _element_list[iter]._massList.size(); b++)
		{
			_element_list[iter]._massList[b].Simulate(SystemParams::_dt);
		}

		// IMPOSE CONSTRAINT
		_element_list[iter].UpdateZConstraint();
		for (int b = 0; b < _element_list[iter]._massList.size(); b++)
		{
			_element_list[iter]._massList[b].ImposeConstraints();
		}
	}
}

void StuffWorker::CollisionGrid_PrepareThreadPool()
{
	// prepare vector
	int len = _c_grid_3d->_squares.size();
	int num_threads = SystemParams::_num_threads;
	int thread_stride = (len + num_threads - 1) / num_threads;
	//int half_len = len / 2;


	//std::vector<std::thread> t_list;
	for (int a = 0; a < num_threads; a++)
	{
		int startIdx = a * thread_stride;
		int endIdx = startIdx + thread_stride;
		_my_thread_pool->submit(&CollisionGrid3D::PrecomputeData_Thread, _c_grid_3d, startIdx, endIdx);
	}

	_my_thread_pool->waitFinished();
}

void StuffWorker::Solve_Prepare_Threads()
{
	int len = _element_list.size();
	int num_threads = SystemParams::_num_threads;
	int thread_stride = (len + num_threads - 1) / num_threads;

	std::vector<std::thread> t_list;
	for (int a = 0; a < num_threads; a++)
	{
		int startIdx = a * thread_stride;
		int endIdx = startIdx + thread_stride;
		t_list.push_back(std::thread(&StuffWorker::Solve_Thread, this, startIdx, endIdx));
	}

	for (int a = 0; a < num_threads; a++)
	{
		t_list[a].join();
	}
}

void StuffWorker::Solve_Thread(int startIdx, int endIdx)
{
	for (unsigned int iter = startIdx; iter < endIdx; iter++)
	{
		// make sure...
		if (iter >= _element_list.size()) { break; }

		for (int b = 0; b < _element_list[iter]._massList.size(); b++)
		{
			_element_list[iter]._massList[b].Solve(_containerWorker->_2d_container, _element_list[iter]);
		}
	}
}

void StuffWorker::SolveSprings_Prepare_Threads()
{
	int len = _element_list.size();
	int num_threads = SystemParams::_num_threads;
	int thread_stride = (len + num_threads - 1) / num_threads;

	std::vector<std::thread> t_list;
	for (int a = 0; a < num_threads; a++)
	{
		int startIdx = a * thread_stride;
		int endIdx = startIdx + thread_stride;
		t_list.push_back(std::thread(&StuffWorker::SolveSprings_Thread, this, startIdx, endIdx));
	}

	for (int a = 0; a < num_threads; a++)
	{
		t_list[a].join();
	}
}

void StuffWorker::SolveSprings_Thread(int startIdx, int endIdx)
{
	for (unsigned int iter = startIdx; iter < endIdx; iter++)
	{
		// make sure...
		if (iter >= _element_list.size()) { break; }
		
		_element_list[iter].SolveForSprings3D();
	}
}

void StuffWorker::GetClosestPt_Prepare_Threads()
{
	int len = _element_list.size();
	int num_threads = SystemParams::_num_threads;
	int thread_stride = (len + num_threads - 1) / num_threads;

	std::vector<std::thread> t_list;
	for (int a = 0; a < num_threads; a++)
	{
		int startIdx = a * thread_stride;
		int endIdx = startIdx + thread_stride;
		t_list.push_back(std::thread(&StuffWorker::GetClosestPt_Thread, this, startIdx, endIdx));
	}

	for (int a = 0; a < num_threads; a++)
	{
		t_list[a].join();
	}
}

void StuffWorker::GetClosestPt_Thread(int startIdx, int endIdx)
{
	for (unsigned int iter = startIdx; iter < endIdx; iter++)
	{
		// make sure...
		if (iter >= _element_list.size()) { break; }

		for (int b = 0; b < _element_list[iter]._massList.size(); b++)
		{
			//_element_list[iter]._massList[b].GetClosestPoint5(*StuffWorker::_c_grid_3d, _element_list);
			_element_list[iter]._massList[b].GetClosestPoint5();
		}

	}
}



void StuffWorker::Reset()
{
	//if (_is_paused) { return; }

	//// update closest points
	//for (int a = 0; a < _element_list.size(); a++)
	//{
	//	for (int b = 0; b < _element_list[a]._massList.size(); b++)
	//	{
	//		_element_list[a]._massList[b].Init();
	//	}

	//}
}



void StuffWorker::Solve()
{
	//if (_is_paused) { return; }

	//if(SystemParams::_multithread_test)
	//{
	//	auto start1_c = std::chrono::steady_clock ::now(); // timing
	//	for (int a = 0; a < _element_list.size(); a++)
	//	{
	//		_element_list[a].SolveForSprings3D();
	//	}
	//	auto elapsed1_c = std::chrono::steady_clock ::now() - start1_c; // timing
	//	_springs_cpu_t = std::chrono::duration_cast<std::chrono::microseconds>(elapsed1_c).count(); // timing
	//}
	////Reset();

	//// ~~~~~ T ~~~~~
	//auto start1 = std::chrono::steady_clock::now();
	//SolveSprings_Prepare_Threads();
	//auto elapsed1 = std::chrono::steady_clock::now() - start1; // timing
	//_springs_thread_t = std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count(); // timing
	//// ~~~~~ T ~~~~~

	//if (SystemParams::_multithread_test)
	//{
	//	auto start2_c = std::chrono::steady_clock ::now(); // timing
	//	for (int a = 0; a < _element_list.size(); a++)
	//	{
	//		for (int b = 0; b < _element_list[a]._massList.size(); b++)
	//		{
	//			_element_list[a]._massList[b].Solve(_containerWorker->_2d_container, _element_list[a]);
	//		}
	//	}
	//	auto elapsed2_c = std::chrono::steady_clock ::now() - start2_c; // timing
	//	_solve_cpu_t = std::chrono::duration_cast<std::chrono::microseconds>(elapsed2_c).count(); // timing
	//}

	//auto start2 = std::chrono::steady_clock::now();
	//Solve_Prepare_Threads();
	//auto elapsed2 = std::chrono::steady_clock::now() - start2; // timing
	//_solve_thread_t = std::chrono::duration_cast<std::chrono::microseconds>(elapsed2).count(); // timing
}


void StuffWorker::Simulate()
{
	/*if (_is_paused) { return; }

	for (int a = 0; a < _element_list.size(); a++)
	{
		for (int b = 0; b < _element_list[a]._massList.size(); b++)
		{
			_element_list[a]._massList[b].Simulate(SystemParams::_dt);
		}
	}*/
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
	/*if (_is_paused) { return; }

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
	}*/

	
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
		_element_list[a].UpdateLayerSpringsOgre3D();
		_element_list[a].UpdateAuxSpringsOgre3D();
		_element_list[a].UpdateNegSpaceEdgeOgre3D();
		_element_list[a].UpdateMassListOgre3D();
		_element_list[a].UpdateVelocityMagnitudeOgre3D();
		_element_list[a].UpdateTimeEdgesOgre3D();
		_element_list[a].UpdateClosestSliceOgre3D();
		_element_list[a].UpdateClosestTriOgre3D();
		_element_list[a].UpdateGrowingOgre3D();
		_element_list[a].UpdateCenterOgre3D();
		_element_list[a].UpdateArtsOgre3D();
	}

	StuffWorker::_c_grid_3d->UpdateOgre3D();
	//_element_list[0].UpdateClosestPtsDisplayOgre3D();
	//_element_list[0].UpdateClosestSliceOgre3D();

	_containerWorker->UpdateOgre3D();

}



void StuffWorker::SaveFrames4()
{
	for (int a = 0; a < _element_list.size(); a++)
	{
		std::cout << "(a) elem=" << a << "\n";
		_element_list[a].CalculateLayerTriangles_Drawing();
	}

	AVideoCreator vCreator;
	vCreator.Init(SystemParams::_num_png_frame);

	//for (int l = 0; l < SystemParams::_num_png_frame; l++)
	//{

	float yCenter = SystemParams::_upscaleFactor / 2;
	
	for (int i = 0; i < _element_list.size(); i++)
	{
		std::cout << "(b) elem=" << i << "\n";
		std::vector<std::vector<std::vector<A2DVector>>> per_layer_triangle_drawing = _element_list[i]._per_layer_triangle_drawing;
		for (int l = 0; l < per_layer_triangle_drawing.size(); l++)
		{
			std::vector<std::vector<A2DVector>> triangles_in_a_layer = per_layer_triangle_drawing[l];
			std::vector<std::vector<A2DVector>> arts = UtilityFunctions::FlipY(_element_list[i].GetBilinearInterpolatedArt(triangles_in_a_layer), yCenter);
			vCreator.DrawFilledArt(arts, _element_list[i]._art_b_colors, _element_list[i]._art_f_colors, l);

		}
	}

	std::stringstream ss;
	ss << SystemParams::_save_folder;
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
	MyColor blk_col(0, 0, 0);

	float yCenter = SystemParams::_upscaleFactor / 2;

	for (int i = 0; i < _element_list.size(); i++)
	{
		MyColor col = _element_list[i]._color;

		std::vector<std::vector<std::vector<A2DVector>>> per_layer_triangle_drawing = _element_list[i]._per_layer_triangle_drawing;
		for (int l = 0; l < per_layer_triangle_drawing.size(); l++)
		{
			std::vector<std::vector<A2DVector>> triangles_in_a_layer = per_layer_triangle_drawing[l];
			std::vector<std::vector<A2DVector>> arts = UtilityFunctions::FlipY( _element_list[i].GetBilinearInterpolatedArt(triangles_in_a_layer), yCenter);
			vCreator.DrawFilledArt(arts, col, l);

			for (int a = 0; a < triangles_in_a_layer.size(); a++)
			{
				// iterate triangle
				A2DVector pt1 = UtilityFunctions::FlipY( triangles_in_a_layer[a][0], yCenter);
				A2DVector pt2 = UtilityFunctions::FlipY( triangles_in_a_layer[a][1], yCenter);
				A2DVector pt3 = UtilityFunctions::FlipY( triangles_in_a_layer[a][2], yCenter);

				vCreator.DrawLine(pt1, pt2, blk_col, l);
				vCreator.DrawLine(pt2, pt3, blk_col, l);
				vCreator.DrawLine(pt3, pt1, blk_col, l);
			}			
		}
	}

	std::stringstream ss;
	ss << SystemParams::_save_folder;
	vCreator.Save(ss.str());
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


//void StuffWorker::InitElements(Ogre::SceneManager* scnMgr)
//{	
//	 element files
//	PathIO pathIO;	
//	std::vector<std::string> some_files = pathIO.LoadFiles(SystemParams::_element_folder); ////
//	std::vector<std::vector<std::vector<A2DVector>>> art_paths;
//	for (unsigned int a = 0; a < some_files.size(); a++)
//	{
//		 is path valid?
//		if (some_files[a] == "." || some_files[a] == "..") { continue; }
//		if (!UtilityFunctions::HasEnding(some_files[a], ".path")) { continue; }
//
//		art_paths.push_back(pathIO.LoadElement(SystemParams::_element_folder + some_files[a]));
//	}
//
//	int elem_iter = 0;
//	int elem_sz = art_paths.size();
//	float initialScale = SystemParams::_element_initial_scale; // 0.05
//
//	 docking
//	A2DVector startPt(80, 80);
//	A2DVector endPt(345, 345);
//	/*A2DVector startPt(420, 80);
//	A2DVector endPt(165, 345);
//	{
//		int idx = _element_list.size();
//		AnElement elem;
//		elem.Triangularization(art_paths[elem_iter++ % elem_sz], idx);
//		elem.ComputeBary();
//		elem.ScaleXY(initialScale);
//		
//		
//		elem.TranslateXY(startPt.x, startPt.y);
//		elem.DockEnds(startPt, endPt);
//		
//		elem.CalculateRestStructure();
//		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
//		elem.InitMeshOgre3D(scnMgr, pNode, "StarTube" + std::to_string(idx), "Examples/TransparentTest2");
//		_element_list.push_back(elem);
//	}*/
//
//	for (int a = 0; a < _containerWorker->_randomPositions.size(); a++)
//	{
//		if (UtilityFunctions::DistanceToFiniteLine(startPt, endPt, _containerWorker->_randomPositions[a]) < 10) { continue; }
//
//		int idx = _element_list.size();
//		AnElement elem;
//		elem.Triangularization(art_paths[elem_iter++ % elem_sz], idx);
//		elem.ComputeBary();
//		 random rotation
//		float radAngle = float(rand() % 628) / 100.0;
//		elem.RotateXY(radAngle);
//
//		elem.ScaleXY(initialScale);
//		elem.TranslateXY(_containerWorker->_randomPositions[a].x, _containerWorker->_randomPositions[a].y);
//
//		elem.CalculateRestStructure();
//		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("TubeNode" + std::to_string(idx));
//		elem.InitMeshOgre3D(scnMgr, pNode, "StarTube" + std::to_string(idx), "Examples/TransparentTest2");
//		_element_list.push_back(elem);		
//
//		if (_element_list.size() == SystemParams::_num_element_pos_limit) { break; }
//
//		std::mt19937 g(SystemParams::_seed);
//		std::shuffle(_containerWorker->_randomPositions.begin(), _containerWorker->_randomPositions.end(), g);
//	}
//	
//	 ----- Collision grid 3D -----
//	StuffWorker::_c_grid_3d->Init();
//	StuffWorker::_c_grid_3d->InitOgre3D(scnMgr);
//	 ---------- Assign to collision grid 3D ----------
//	for (unsigned int a = 0; a < _element_list.size(); a++)
//	{
//		 time triangle
//		for (unsigned int b = 0; b < _element_list[a]._surfaceTriangles.size(); b++)
//		{
//			AnIdxTriangle tri = _element_list[a]._surfaceTriangles[b];
//			A3DVector p1      = _element_list[a]._massList[tri.idx0]._pos;
//			A3DVector p2      = _element_list[a]._massList[tri.idx1]._pos;
//			A3DVector p3      = _element_list[a]._massList[tri.idx2]._pos;
//
//			_c_grid_3d->InsertAPoint( (p1._x + p2._x + p3._x) * 0.333,
//				                      (p1._y + p2._y + p3._y) * 0.333,
//				                      (p1._z + p2._z + p3._z) * 0.333,
//				                      a,  // which element
//				                      b); // which triangle		
//		}
//
//		 assign
//		for (unsigned int b = 0; b < _element_list[a]._massList.size(); b++)
//		{
//			_element_list[a]._massList[b]._c_grid_3d = _c_grid_3d; // assign grid to mass
//		}
//	}
//
//	 ---------- Calculate num vertex ----------
//	_num_vertex = 0;
//	for (unsigned int a = 0; a < _element_list.size(); a++)
//	{
//		_num_vertex += _element_list[a]._massList.size();
//	}
//
//	 ----- Interpolation collision grid -----
//	 INTERP WONT WORK BECAUSE OF THIS
//	for (int a = 0; a < SystemParams::_interpolation_factor; a++)
//		{ _interp_c_grid_list.push_back(new CollisionGrid2D); }
//
//	 ----- Assign to interpolation collision grid -----
//	/*for (int a = 0; a < _element_list.size(); a++)
//	{
//		for (int b = 0; b < _element_list[a]._interp_massList.size(); b++)
//		{
//			int c_grid_idx = _element_list[a]._interp_massList[b]._layer_idx;
//			A3DVector p1 = _element_list[a]._interp_massList[b]._pos;
//
//			_interp_c_grid_list[c_grid_idx]->InsertAPoint(p1._x, p1._y, a, b); // assign mass to grid			
//			_element_list[a]._interp_massList[b]._c_grid = _interp_c_grid_list[c_grid_idx]; // assign grid to mass
//		}
//	}*/
//	 INTERP WONT WORK BECAUSE OF THIS
//
//	 debug delete me
//	/*std::vector<A3DVector> tri1;
//	tri1.push_back(A3DVector(300, 0, -100));
//	tri1.push_back(A3DVector(400, 0, -400));
//	tri1.push_back(A3DVector(0, 0, -10));
//	
//	
//	_triangles.push_back(tri1);
//
//	 material
//	Ogre::MaterialPtr line_material = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("TriDebugLines");
//	line_material->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(1, 0, 0, 1));
//	_debug_lines_tri = new DynamicLines(line_material, Ogre::RenderOperation::OT_LINE_LIST);
//	for (int a = 0; a < _triangles.size(); a++)
//	{
//		A3DVector pt1 = _triangles[a][0];
//		A3DVector pt2 = _triangles[a][1];
//		A3DVector pt3 = _triangles[a][2];
//
//		_debug_lines_tri->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
//		_debug_lines_tri->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
//
//		_debug_lines_tri->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
//		_debug_lines_tri->addPoint(Ogre::Vector3(pt3._x, pt3._y, pt3._z));
//
//		_debug_lines_tri->addPoint(Ogre::Vector3(pt3._x, pt3._y, pt3._z));
//		_debug_lines_tri->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
//	}
//
//
//	for (int a = 0; a < 100; a++)
//	{
//		A3DVector randPt(rand()% 500, rand() % 500, -(rand() % 500));
//		A3DVector closestPt = UtilityFunctions::ClosestPointOnTriangle2(randPt, _triangles[0][0], _triangles[0][1], _triangles[0][2]);
//
//		_debug_lines_tri->addPoint(Ogre::Vector3(randPt._x, randPt._y, randPt._z));
//		_debug_lines_tri->addPoint(Ogre::Vector3(closestPt._x, closestPt._y, closestPt._z));
//
//	}
//
//	_debug_lines_tri->update();
//	_debugNode_tri = scnMgr->getRootSceneNode()->createChildSceneNode("debug_lines_tri_debug");
//	_debugNode_tri->attachObject(_debug_lines_tri);*/
//
//}

//void StuffWorker::Interp_SaveFrames()
//{
	//int l = StuffWorker::_interpolation_iter;
	/*for (int a = 0; a < _element_list.size(); a++)
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
	}*/
	//}

/*void StuffWorker::SaveFrames2()
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
}*/


/*void StuffWorker::SaveFrames()
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
}*/

//void StuffWorker::EnableInterpolationMode()
//{
	/*std::cout << "enable interpolation\n";

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
	}*/
	//	
	//}

	//void StuffWorker::DisableInterpolationMode()
	//{
	//	std::cout << "disable interpolation\n";
	//
	//	StuffWorker::_interp_mode  = false;
	//	StuffWorker::_interp_iter  = 0;
	////	StuffWorker::_interpolation_value = 0;
	////
	////	for (int a = 0; a < _element_list.size(); a++)
	////	{
	////		_element_list[a].DisableInterpolationMode();
	////	}
	//}


//

// INTERPOLATION
//void StuffWorker::Interp_Update()
//{
	/*// ----- for closest point calculation -----
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
	}*/

	//}


//void StuffWorker::Interp_Reset()
//{
//	// update closest points
//	/*for (int a = 0; a < _element_list.size(); a++)
//	{
//		for (int b = 0; b < _element_list[a]._interp_massList.size(); b++)
//		{
//			_element_list[a]._interp_massList[b].Init();
//		}
//
//	}*/
//}

//void StuffWorker::Interp_Solve()
//{
//	/*for (int a = 0; a < _element_list.size(); a++)
//	{
//		_element_list[a].Interp_SolveForSprings2D();
//
//		for (int b = 0; b < _element_list[a]._interp_massList.size(); b++)
//		{
//			_element_list[a]._interp_massList[b].Solve(_containerWorker->_2d_container);
//		}
//	}*/
//}

//void StuffWorker::Interp_Simulate()
//{
//	/*for (int a = 0; a < _element_list.size(); a++)
//	{
//		for (int b = 0; b < _element_list[a]._interp_massList.size(); b++)
//		{
//			_element_list[a]._interp_massList[b].Interp_Simulate(SystemParams::_dt);
//		}
//	}*/
//}
//
//bool StuffWorker::Interp_HasOverlap()
//{
//	/*for (int a = 0; a < _element_list.size(); a++)
//	{
//		if (_element_list[a].Interp_HasOverlap())
//			return true;
//	}*/
//	return false;
//}
