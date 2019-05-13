#ifndef _STUFF_WORKER_H_
#define _STUFF_WORKER_H_

#include "AnElement.h"
#include "CollisionGrid2D.h"

//#include "ContainerWorker.h"

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
	void Interp_Simulate();     // (non-velocity verlet) iterate the masses by the change in time	
	void Interp_ImposeConstraints();

	void Update();
	void Reset();          // reset forces to zero
	void Solve();            // calculate forces
	void Simulate();     // (non-velocity verlet) iterate the masses by the change in time	
	void ImposeConstraints();
	void UpdateOgre3D();

	void SaveFrames();

public:
	static bool  _interpolation_mode;
	static int   _interpolation_iter; // from zero to _interpolation_factor - 1
	//static float _interpolation_value;
	static std::vector<CollisionGrid2D*> _interp_c_grid_list;


	void EnableInterpolationMode();
	void DisableInterpolationMode();

public:

	ContainerWorker* _containerWorker;

	//AnElement* _elem;
	static std::vector<AnElement> _element_list;

	// collission grid
	static std::vector<CollisionGrid2D*> _c_grid_list;
};

#endif