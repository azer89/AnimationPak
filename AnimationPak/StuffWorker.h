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

	void InitElements(Ogre::SceneManager* scnMgr);
	/*void CreateRandomElementPoints(std::vector<A2DVector> ornamentBoundary,
									float img_length,
									std::vector<A2DVector>& randomPoints,
									int& boundaryPointNum);*/

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

public:
	static bool  _interp_mode;
	static int   _interp_iter; // from zero to _interpolation_factor - 1
	static std::vector<CollisionGrid2D*> _interp_c_grid_list;


	void EnableInterpolationMode();
	void DisableInterpolationMode();

	//void LoadElements();

public:

	ContainerWorker* _containerWorker;

	static std::vector<AnElement> _element_list;

	//static PoissonGenerator::DefaultPRNG _PRNG;

	
	//static std::vector<CollisionGrid2D*> _c_grid_list; // collission grid 2D
	static CollisionGrid3D* _c_grid_3d; // collission grid 3D

	AVideoCreator _video_creator;

public:
	
	// points-triangles debug
	/*std::vector<std::vector<A3DVector>> _triangles;

	// points-triangles debug
	DynamicLines*    _debug_lines_tri;
	Ogre::SceneNode* _debugNode_tri;
	*/
};

#endif