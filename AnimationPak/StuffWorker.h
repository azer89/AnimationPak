#ifndef _STUFF_WORKER_H_
#define _STUFF_WORKER_H_

#include "AnElement.h"
#include "CollisionGrid2D.h"

#include <vector>

class StuffWorker
{
public:
	StuffWorker();

	~StuffWorker();

	void InitElements(Ogre::SceneManager* scnMgr);

	void Update();
	void Reset();          // reset forces to zero
	void Solve();            // calculate forces
	void Simulate();     // (non-velocity verlet) iterate the masses by the change in time	
	void UpdateViz();

public:

	//AnElement* _elem;
	static std::vector<AnElement> _element_list;

	// collission grid
	static std::vector<CollisionGrid2D*> _c_grid_list;
};

#endif