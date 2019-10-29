#ifndef _STUFF_WORKER_H_
#define _STUFF_WORKER_H_

#include "AnElement.h"
#include "CollisionGrid2D.h"
#include "CollisionGrid3D.h"
#include "AVideoCreator.h"

//#include "PoissonGenerator.h"
//#include "ContainerWorker.h"

#include <vector>

class ContainerWorker;

class StuffWorker
{
public:
	StuffWorker();

	~StuffWorker();

	//void InitElements(Ogre::SceneManager* scnMgr);
	void InitElements2(Ogre::SceneManager* scnMgr);
	void InitElements_TwoMovingElements(Ogre::SceneManager* scnMgr); // SCENE
	void InitElements_OneMovingElement(Ogre::SceneManager* scnMgr);

	//void Interp_Update();
	//void Interp_Reset();          // reset forces to zero
	//void Interp_Solve();            // calculate forces
	//void Interp_Simulate();     // 
	//bool Interp_HasOverlap();

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

	static CollisionGrid3D* _c_grid_3d; // collission grid 3D
	//static CollisionGrid3D _c_grid_3d;

	//AVideoCreator _video_creator;

	//float _micro_1_thread;
	//float _micro_n_thread;
	int _cg_thread_t;
	int _springs_thread_t;
	int _c_pt_thread_t;
	int _solve_thread_t;

	int _cg_cpu_t;
	int _springs_cpu_t;
	int _c_pt_cpu_t;
	int _solve_cpu_t;

	int _max_c_pts;
	int _max_c_pts_approx;


public:
};

#endif