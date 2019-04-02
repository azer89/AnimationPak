
#ifndef __A_MASS_H__
#define __A_MASS_H__

#include "A3DVector.h"
#include "AMass.h"

#include "CollisionGrid2D.h"

class AMass
{
public:

	float   _mass;    // is likely only one
	A3DVector _pos;	  // current
	A3DVector _velocity;

	int _self_idx; // for identification
	int _parent_idx; // parent identification
	int _debug_which_layer; // 0 1 2 3 4 5

	CollisionGrid2D* _c_grid;



public:
	AMass();

	~AMass();

	// Constructor
	AMass(float x, float y, float z);

	// Constructor
	// x y z mass_idx element_idx layer_idx
	AMass(float x, 
		  float y, 
		  float z, 
		  int self_idx, 
		  int parent_idx, 
		  int debug_which_layer);

	// Constructor
	AMass(A3DVector pos);

	void CallMeFromConstructor();

	void Init(); // reset forces to zero
	void Simulate(float dt);
	void Solve(/*Need more parameters*/);

	void GetClosestPoint();

public:
	A3DVector _edgeForce;
	A3DVector _repulsionForce;
	A3DVector _boundaryForce;
	A3DVector _overlapForce;
	A3DVector _rotationForce;


	float                _closestDist;
	std::vector<int>     _closestGraphIndices;
	
	std::vector<A2DVector> _closestPoints;
	int _closestPt_actual_sz; // reserve for avoiding push_back, actual size of the vector
	int _closestPt_fill_sz;   // reserve for avoiding push_back, filled size of the vector

	bool _is_inside;
};

#endif