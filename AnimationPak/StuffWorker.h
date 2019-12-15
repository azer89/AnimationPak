#ifndef _STUFF_WORKER_H_
#define _STUFF_WORKER_H_

#include "AnElement.h"
#include "CollisionGrid2D.h"
#include "CollisionGrid3D.h"
#include "AVideoCreator.h"

//#include "PoissonGenerator.h"
//#include "ContainerWorker.h"

#include <vector>

#include "ThreadPool.h"
#include "ATimerStat.h"
#include "TubeConnector.h"

class ContainerWorker;

class StuffWorker
{
public:
	StuffWorker();

	~StuffWorker();

	//void InitElements(Ogre::SceneManager* scnMgr);
	void DockElementsOnPaths(std::vector <std::vector<A3DVector>> paths, 
		              std::vector<std::vector<int>> layer_indices,
		              std::vector<AnElement> temp_elements,
					  Ogre::SceneManager* scnMgr);

	void JitterPosAndRotation(float pos_max_offset, A2DVector& pos_offset, float& rot_val);

	void InitElementsAndCGrid(Ogre::SceneManager* scnMgr);
	void InitElements_TwoMovingElements(Ogre::SceneManager* scnMgr); // SCENE
	void InitElements_OneMovingElement(Ogre::SceneManager* scnMgr);
	void InitAnimated_Elements(Ogre::SceneManager* scnMgr);
	void InitStar_Elements(Ogre::SceneManager* scnMgr);

	void InitSavedScenes(Ogre::SceneManager* scnMgr); // for testing purpose

	void ConnectTubeEnds();

	//void Interp_Update();
	//void Interp_Reset();          // reset forces to zero
	//void Interp_Solve();            // calculate forces
	//void Interp_Simulate();     // 
	//bool Interp_HasOverlap();

	void AlmostAllUrShit();
	void AlmostAllUrShit_SingleThread();
	void AlmostAllUrShit_PrepareThreadPool();
	void AlmostAllUrShit_ThreadTask(int startIdx, int endIdx);
	void CollisionGrid_PrepareThreadPool();

	void SaveScene();
	void SaveStatistics();
	int StillGrowing();


	void Update();
	void Reset();          // reset forces to zero
	void Solve();            // calculate forces
	void Simulate();     // 
	void ImposeConstraints();
	void UpdateOgre3D();

	void GetClosestPt_Prepare_Threads();
	void GetClosestPt_Thread(int startIdx, int endIdx);

	void SolveSprings_Prepare_Threads();
	void SolveSprings_Thread(int startIdx, int endIdx);

	void Solve_Prepare_Threads();
	void Solve_Thread(int startIdx, int endIdx);

	

	//void Interp_SaveFrames();
	//void SaveFrames();
	//void SaveFrames2();
	void SaveFrames3();
	void SaveFrames4();

public:
	bool _is_paused;

	//static bool  _interp_mode;
	//static int   _interp_iter; // from zero to _interpolation_factor - 1
	//static std::vector<CollisionGrid2D*> _interp_c_grid_list;


	//void EnableInterpolationMode();
	//void DisableInterpolationMode();

public:

	int _num_vertex;

	ContainerWorker* _containerWorker;

	static std::vector<AnElement> _element_list;

	//std::vector<TubeConnector> _tube_connectors;

	static CollisionGrid3D* _c_grid_3d; // collission grid 3D
	//static CollisionGrid3D _c_grid_3d;

	//AVideoCreator _video_creator;
	ThreadPool* _my_thread_pool;

	//float _micro_1_thread;
	//float _micro_n_thread;
	
	//int _springs_thread_t;
	//int _c_pt_thread_t;
	//int _solve_thread_t;

	ATimerStat _almostall_multi_t;
	ATimerStat _almostall_single_t;

	ATimerStat _cg_multi_t;
	ATimerStat _cg_single_t;
	//int _springs_cpu_t;
	//int _c_pt_cpu_t;
	//int _solve_cpu_t;

	// additional
	ATimerStat _cg_move_points;

	int _max_c_pts;
	int _max_c_pts_approx;

	int _num_iteration;


public:
};

#endif