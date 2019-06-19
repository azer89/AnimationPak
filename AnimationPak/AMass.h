
#ifndef __A_MASS_H__
#define __A_MASS_H__

#include "A3DVector.h"
#include "A3DObject.h"
#include "AMass.h"

#include "AnIdxTriangle.h"

#include "CollisionGrid2D.h"
#include "CollisionGrid3D.h"

class AMass
{
public:

	float   _mass;    // is likely only one
	A3DVector _pos;	  // current
	A3DVector _velocity;
	
	int _self_idx; // for identification, not PER LAYER
	int _parent_idx; // parent identification
	int _layer_idx; // layer idx

	//CollisionGrid2D* _c_grid;
	CollisionGrid3D* _c_grid_3d;

	//bool _isInside;

	bool _isDocked;
	bool _is_inside;
	std::vector<A2DVector> _closest_boundary_slice;
	bool _is_boundary;

	A3DVector _dockPoint; // you probably want the dockpoint be 2D?

	std::vector<AnIdxTriangle> _triangles; // for overlap force

	std::vector<AnIdxTriangle>    _timeTriangles; // for 3D collision grid

	float                _closestDist; // for stop growing??? need to check

public: // need to be public for debugging purpose
	//int _closestPt_actual_sz; // reserve for avoiding push_back, actual size of the vector
	//int _closestPt_fill_sz;   // reserve for avoiding push_back, filled size of the vector
	//std::vector<A2DVector> _closestPoints;


	std::vector<A3DVector> _closestPoints3D;
	//A3DVector* _closestPoints3D;
	int _closest_pd_actual_len;
	int _closest_pd_max_len;

private:		
	std::vector<int>     _closestGraphIndices;

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
		  int debug_which_layer,
		  bool is_boundary = false);

	// Constructor
	AMass(A3DVector pos);

	void CallMeFromConstructor();

	void Init(); // reset forces to zero
	void Simulate(float dt);
	void Interp_Simulate(float dt); // debug delete me...
	void Solve(const std::vector<A2DVector>& container);

	void ImposeConstraints();

	//void GetClosestPoint();

	//void GetClosestPoint2();

	//void GetClosestPoint3();

	void GetClosestPoint4();

	void Interp_GetClosestPoint();

	A3DVector GetClosestPtFromArray(int elem_idx, std::vector<A3DObject>& tempClosestObj3D);

	void Grow(float growth_scale_iter, float dt);

	void Print()
	{
		std::cout << _self_idx << " - " << _parent_idx << "\n";
	}

public:
	A3DVector _edgeForce;
	A3DVector _repulsionForce;
	A3DVector _boundaryForce;
	A3DVector _overlapForce;
	A3DVector _rotationForce;


	
};

#endif