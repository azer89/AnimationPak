
#ifndef __A_MASS_H__
#define __A_MASS_H__

#include "A3DVector.h"
#include "A3DObject.h"
#include "AMass.h"

#include "AnIdxTriangle.h"

#include "CollisionGrid2D.h"
#include "CollisionGrid3D.h"

class AnElement;

class AMass
{
public:

	float     _mass;    // is likely only one
	float     _ori_z_pos;
	A3DVector _pos;	  // current
	A3DVector _velocity;
	
	int _self_idx; // for identification, not PER LAYER
	int _parent_idx; // parent identification
	int _layer_idx; // layer idx
	//CollisionGrid3D* _c_grid_3d;

	//bool _isInside;

	bool _isDocked;
	bool _is_inside;
	std::vector<A3DVector> _closest_boundary_slice;
	//std::vector<std::vector<A3DVector>> _closest_tri_array; // currently disabled!
	bool _is_boundary;

	A3DVector _dockPoint; // you probably want the dockpoint be 2D?

	std::vector<AnIdxTriangle> _triangles; // for overlap force

	//float _closestDist; // for stop growing??? need to check

public:
	int _closest_elem_idx;

	// closest points (exact measurements)
	std::vector<A3DVector> _c_pts;	
	int _c_pts_fill_size;	
	int _c_pts_max_size;

	// closest points (approx measurement)
	std::vector<std::pair<A3DVector, int>> _c_pts_approx; // barnes hut: (1) center of square (2) num points in that square
	int _c_pts_approx_fill_size;
	int _c_pts_approx_max_size;

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

	void ResetForces(); // reset forces to zero
	void Simulate(float dt);
	void Interp_Simulate(float dt); // debug delete me...
	void Solve(const std::vector<A2DVector>& container, AnElement& parentElem);

	void ImposeConstraints();

	void GetClosestPoint4();

	void Interp_GetClosestPoint();

	//A3DVector GetClosestPtFromArray(int elem_idx, std::vector<A3DObject>& tempClosestObj3D);

	void Grow(float growth_scale_iter, float dt);

	void Print()
	{
		std::cout << _self_idx << " - " << _parent_idx << "\n";
	}

public:
	//A3DVector _edgeForce_cuda;
	A3DVector _edgeForce;
	A3DVector _zForce;
	A3DVector _repulsionForce;
	A3DVector _boundaryForce;
	A3DVector _overlapForce;
	A3DVector _rotationForce;
	A3DVector _dockForce;
};

#endif