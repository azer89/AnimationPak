#ifndef _STUFF_WORKER_H_
#define _STUFF_WORKER_H_

#include "AnElement.h"
#include "CollisionGrid2D.h"
#include "CollisionGrid3D.h"
#include "AVideoCreator.h"

#include "CUDAWorker.cuh"

#include <vector>

class ContainerWorker;

class StuffWorker
{
public:
	StuffWorker();

	~StuffWorker();

	void InitElements(Ogre::SceneManager* scnMgr);

	void Interp_Update();
	void Interp_Reset();          // reset forces to zero
	void Interp_Solve();            // calculate forces
	void Interp_Simulate();     // 
	bool Interp_HasOverlap();

	void Update();
	void Reset();          // reset forces to zero
	void Solve();            // calculate forces
	void Simulate();     // 
	void ImposeConstraints();
	void UpdateOgre3D();

	void Interp_SaveFrames();
	void SaveFrames();
	void SaveFrames2();
	void SaveFrames3();
	void SaveFrames4();

public:
	bool _is_paused;

	static bool  _interp_mode;
	static int   _interp_iter; // from zero to _interpolation_factor - 1
	static std::vector<CollisionGrid2D*> _interp_c_grid_list;


	void EnableInterpolationMode();
	void DisableInterpolationMode();

public:

	int _num_vertex;
	int _num_spring;

	ContainerWorker* _containerWorker;

	static std::vector<AnElement> _element_list;

	static CollisionGrid3D* _c_grid_3d; // collission grid 3D

	AVideoCreator _video_creator;

	CUDAWorker* _cu_worker;

public:
};

#endif