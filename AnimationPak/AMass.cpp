
#include "AMass.h"

#include "SystemParams.h"

#include "StuffWorker.h"

#include "UtilityFunctions.h"

AMass::AMass()
{
	//this->_closestPoints3D = 0;
	//this->_m     = 0;             // mass is always one
	this->_pos = A3DVector(0, 0, 0);
	this->_self_idx = -1;
	//this->_cellIdx = -1;

	CallMeFromConstructor();
}

// Constructor
AMass::AMass(float x, float y, float z)
{
	//this->_closestPoints3D = 0;
	this->_pos = A3DVector(x, y, z);
	this->_self_idx = -1;

	CallMeFromConstructor();
}

// Constructor
AMass::AMass(float x, 
	         float y, 
	         float z, 
	         int self_idx, 
	         int parent_idx, 
	         int debug_which_layer,
			 bool is_boundary)
{
	//this->_closestPoints3D = 0;
	this->_pos               = A3DVector(x, y, z);
	this->_layer_idx = debug_which_layer;
	this->_self_idx          = self_idx;
	this->_parent_idx        = parent_idx;
	this->_is_boundary       = is_boundary;

	CallMeFromConstructor();
}


// Constructor
AMass::AMass(A3DVector pos)
{
	//this->_closestPoints3D = 0;
	this->_pos = pos;
	this->_self_idx = -1;

	CallMeFromConstructor();
}

AMass::~AMass()
{
	/*if(_closestPoints3D)
	{
		delete[] _closestPoints3D;
	}*/
}

void  AMass::PrepareCPtsArrays() // see _c_pts and _c_pts_approx
{
	_c_pts_fill_size = 0;
	_c_pts_approx_fill_size = 0;
	_c_pts_max_size = 0;
	_c_pts_approx_max_size = 0;

	if (_is_boundary)
	{
		_c_pts_max_size = SystemParams::_max_exact_array_len; // to do: rename _max_exact_array_len
		_c_pts = std::vector<A3DVector>(_c_pts_max_size, A3DVector(0, 0, 0));

		_c_pts_approx_max_size = SystemParams::_max_exact_array_len; // to do: rename _max_exact_array_len
		_c_pts_approx = std::vector<std::pair<A3DVector, int>>(_c_pts_approx_max_size, std::pair<A3DVector, int>(A3DVector(0, 0, 0), 0)); // very complicated
	}
}

void AMass::CallMeFromConstructor()
{
	_closest_elem_idx = -1;

	_ori_z_pos = _pos._z;

	_valence = 0;

	_velocity = A3DVector(0, 0, 0);

	// hard parameter for closest pt search
	//_closestPt_fill_sz = 0;
	//_closestPt_actual_sz = 150; // BE CAREFUL HARD CODED PARAM!!!
	//_closestPoints = std::vector <A2DVector>(_closestPt_actual_sz);

	_is_inside = false;

	_closest_dist = 10000000000;

	_isDocked = false;

	//_c_pts_fill_size = 0;
	//_c_pts_max_size = SystemParams::_max_exact_array_len; // to do: rename _max_exact_array_len
	//_c_pts = std::vector<A3DVector>(_c_pts_max_size, A3DVector(0,0,0));

	//_c_pts_approx_fill_size = 0;
	//_c_pts_approx_max_size = SystemParams::_max_exact_array_len; // to do: rename _max_exact_array_len
	//_c_pts_approx = std::vector<std::pair<A3DVector, int>>(_c_pts_approx_max_size, std::pair<A3DVector, int>(A3DVector(0, 0, 0), 0)); // very complicated

	Init();
}

void AMass::Init()
{
	//_attractionForce = AVector(0, 0);
	this->_edgeForce      = A3DVector(0, 0, 0);
	this->_zForce         = A3DVector(0, 0, 0);
	this->_repulsionForce = A3DVector(0, 0, 0);
	this->_boundaryForce  = A3DVector(0, 0, 0);
	this->_overlapForce   = A3DVector(0, 0, 0);
	this->_rotationForce  = A3DVector(0, 0, 0);
}

// debug delete me
void AMass::Interp_Simulate(float dt)
{
	_velocity += ((_edgeForce +
		_zForce +
		_repulsionForce +
		_boundaryForce +
		_overlapForce +
		_rotationForce) * dt);
	float len = _velocity.Length();

	float capVal = SystemParams::_velocity_cap * dt;

	if (len > capVal)
	{
		_velocity = _velocity.Norm() * capVal;
	}

	_pos = _pos + _velocity * dt;
}

void AMass::Simulate(float dt)
{
	//if (_isDocked) { return; }

	// oiler
	_velocity += ((_edgeForce +
		_zForce +
		_repulsionForce +
		_boundaryForce +
		_overlapForce +
		_rotationForce) * dt);
	float len = _velocity.Length();

	float capVal = SystemParams::_velocity_cap * dt;

	if (len > capVal)
	{
		_velocity = _velocity.Norm() * capVal;
	}

	_pos = _pos + _velocity * dt;
}

void AMass::ImposeConstraints()
{
	if (_layer_idx == 0)
	{
		_pos._z = 0;
	}
	else if (_layer_idx == SystemParams::_num_layer - 1)
	{
		_pos._z = -SystemParams::_upscaleFactor;
	}

	// boundary
	/*if (_pos._x < 0) { _pos._x = 0; }
	if (_pos._x >= SystemParams::_upscaleFactor) { _pos._x = SystemParams::_upscaleFactor - 1; }
	
	if (_pos._y < 0) { _pos._y = 0; }
	if (_pos._y >= SystemParams::_upscaleFactor) { _pos._y = SystemParams::_upscaleFactor - 1; }*/
}

void AMass::Interp_GetClosestPoint()
{
	/*
	if (!_is_boundary) { return; }
	if (_parent_idx < 0 || _parent_idx >= StuffWorker::_element_list.size()) { return; }

	this->_closestGraphIndices.clear();
	this->_closestPt_fill_sz = 0;
	this->_is_inside = false;           // "inside" flag

	_c_grid->GetGraphIndices2B(_pos._x, _pos._y, _closestGraphIndices);

	if (_closestGraphIndices.size() > 0)
	{
		//std::cout << "closestpt";
		std::vector<bool> insideGraphFlags;
		int sz = _closestGraphIndices.size();
		for (unsigned int a = 0; a < sz; a++)
		{
			// uncomment me
			if (_closestGraphIndices[a] == _parent_idx) { insideGraphFlags.push_back(true); continue; }

			if (UtilityFunctions::InsidePolygon(StuffWorker::_element_list[_closestGraphIndices[a]]._interp_per_layer_boundary[_layer_idx], _pos._x, _pos._y))
			{
				insideGraphFlags.push_back(true);
				_is_inside = true;
				continue; // can be more than one
			}
			else
			{
				insideGraphFlags.push_back(false);
			}
		}

		// closest pts
		int sz2 = sz;
		if (sz2 > _closestPt_actual_sz) { sz2 = _closestPt_actual_sz; }  // _closestPt_actual_sz --> BE CAREFUL HARD CODED PARAM!!!
		for (unsigned int a = 0; a < sz2; a++)
		{
			if (insideGraphFlags[a]) { continue; }

			// the only difference from AMass::GetClosestPoint()
			A2DVector pt = StuffWorker::_element_list[_closestGraphIndices[a]].Interp_ClosestPtOnALayer(_pos.GetA2DVector(), _layer_idx);
			_closestPoints[_closestPt_fill_sz++] = pt;
		}
	}

	// this is used in AGraph
	_closestDist = std::numeric_limits<float>::max();
	for (unsigned int a = 0; a < _closestPt_fill_sz; a++)
	{
		float d = _closestPoints[a].DistanceSquared(_pos.GetA2DVector());  // 2D!!!! // SQUARED!!!
		if (d < _closestDist)
		{
			_closestDist = d;
		}
	}
	_closestDist = std::sqrt(_closestDist); // SQRT
	*/
}

/*
A3DVector AMass::GetClosestPtFromArray(int elem_idx, std::vector<A3DObject>& tempClosestObj3D)
{
	float dist = 10000000;


	for (int a = 0; a < tempClosestObj3D.size(); a++)
	{
		if (tempClosestObj3D[a]._info1 != elem_idx) { continue; }
	}
}*/

void AMass::GetClosestPoint5(const CollisionGrid3D& c_grid, const std::vector<AnElement>& element_list)
{
	if (!_is_boundary) { return; }
	//if (_parent_idx < 0 || _parent_idx >= StuffWorker::_element_list.size()) { return; } // why???

	this->_is_inside = false;           // "inside" flag

	// clear
	_c_pts_fill_size = 0;
	_c_pts_approx_fill_size = 0;
	_closest_elem_idx = -1;

	//_closest_tri_array.clear(); // disabled :(

	int square_idx = c_grid.GetSquareIndexFromFloat(_pos._x, _pos._y, _pos._z);

	//for (unsigned int a = 0; a < exact_pd.size(); a++)
	_closest_dist = 10000000000;
	float closest_tri_idx = -1;
	A3DSquare* sq = c_grid._squares[square_idx];

	// ----- exact closest point -----
	for (unsigned int a = 0; a < sq->_c_pt_fill_size; a++)
	{
		if (sq->_c_pt[a].first == _parent_idx) { continue; }
		A3DVector pt = element_list[sq->_c_pt[a].first].ClosestPtOnATriSurface_Const(sq->_c_pt[a].second, _pos);

		_c_pts[_c_pts_fill_size++] = pt;

		// closest element
		float distSq = pt.DistanceSquared(_pos);
		if (distSq < _closest_dist)
		{
			_closest_dist = distSq;
			_closest_elem_idx = sq->_c_pt[a].first;
			closest_tri_idx = sq->_c_pt[a].second;
		}
	}

	// ----- inside outside -----
	if (_closest_elem_idx != -1)
	{
		int layer_idx = StuffWorker::_element_list[_closest_elem_idx]._surfaceTriangles[closest_tri_idx]._layer_idx; // original
		_is_inside = StuffWorker::_element_list[_closest_elem_idx].IsInside_Const(layer_idx, _pos, _closest_boundary_slice);
	}

	// ----- approx closest point -----
	int current_sq_idx = -1;
	for (unsigned int a = 0; a < sq->_c_pt_approx_fill_size; a++)
	{
		int p_idx = sq->_c_pt_approx[a].first;
		if (p_idx == _parent_idx) { continue; }

		int temp_sq_idx = sq->_c_pt_approx[a].second; // square idx, assume they're on the same segments.

		if (temp_sq_idx == current_sq_idx) // same
		{
			_c_pts_approx[_c_pts_approx_fill_size].second++;
		}
		else // new one
		{
			current_sq_idx = temp_sq_idx;

			_c_pts_approx[_c_pts_approx_fill_size].first = A3DVector(c_grid._squares[current_sq_idx]->_xCenter,
				c_grid._squares[current_sq_idx]->_yCenter,
				-c_grid._squares[current_sq_idx]->_zCenter); // negative
			_c_pts_approx[_c_pts_approx_fill_size].second = 1;
			_c_pts_approx_fill_size++;
		}
	}
}

void AMass::GetClosestPoint4()
{
	if (!_is_boundary) { return; }
	//if (_parent_idx < 0 || _parent_idx >= StuffWorker::_element_list.size()) { return; } // why???

	this->_is_inside = false;           // "inside" flag

	// clear
	_c_pts_fill_size = 0;
	_c_pts_approx_fill_size = 0;
	_closest_elem_idx = -1;

	//_closest_tri_array.clear(); // disabled :(

	int square_idx = StuffWorker::_c_grid_3d->GetSquareIndexFromFloat(_pos._x, _pos._y, _pos._z);

	//for (unsigned int a = 0; a < exact_pd.size(); a++)
	float closest_dist = 10000000000;	
	float closest_tri_idx = -1;
	A3DSquare* sq = StuffWorker::_c_grid_3d->_squares[square_idx];

	// ----- exact closest point -----
	for (unsigned int a = 0; a < sq->_c_pt_fill_size; a++)
	{
		if (sq->_c_pt[a].first == _parent_idx) { continue; }
		A3DVector pt = StuffWorker::_element_list[sq->_c_pt[a].first].ClosestPtOnATriSurface(sq->_c_pt[a].second, _pos);

		// disabled
		/*
		std::vector<A3DVector> tri(3);
		AnIdxTriangle surface_tri = StuffWorker::_element_list[sq->_c_pt[a].first]._surfaceTriangles[sq->_c_pt[a].second];
		tri[0] = StuffWorker::_element_list[sq->_c_pt[a].first]._massList[surface_tri.idx0]._pos;
		tri[1] = StuffWorker::_element_list[sq->_c_pt[a].first]._massList[surface_tri.idx1]._pos;
		tri[2] = StuffWorker::_element_list[sq->_c_pt[a].first]._massList[surface_tri.idx2]._pos;
		_closest_tri_array.push_back(tri);
		*/

		_c_pts[_c_pts_fill_size++] = pt;

		// closest element
		float distSq = pt.DistanceSquared(_pos);
		if (distSq < closest_dist)
		{
			closest_dist = distSq;
			_closest_elem_idx = sq->_c_pt[a].first;
			closest_tri_idx = sq->_c_pt[a].second;
		}
	}

	// ----- inside outside -----
	if (_closest_elem_idx != -1)
	{		
		int layer_idx = StuffWorker::_element_list[_closest_elem_idx]._surfaceTriangles[closest_tri_idx]._layer_idx; // original
		_is_inside = StuffWorker::_element_list[_closest_elem_idx].IsInside(layer_idx, _pos, _closest_boundary_slice);
	}

	// ----- approx closest point -----
	int current_sq_idx = -1;
	for (unsigned int a = 0; a < sq->_c_pt_approx_fill_size; a++)
	{
		int p_idx = sq->_c_pt_approx[a].first;
		if (p_idx == _parent_idx) { continue; }

		int temp_sq_idx = sq->_c_pt_approx[a].second; // square idx, assume they're on the same segments.

		if (temp_sq_idx == current_sq_idx) // same
		{
			_c_pts_approx[_c_pts_approx_fill_size].second++;
		}
		else // new one
		{
			current_sq_idx = temp_sq_idx;	

			_c_pts_approx[_c_pts_approx_fill_size].first = A3DVector( StuffWorker::_c_grid_3d->_squares[current_sq_idx]->_xCenter,
				                                                      StuffWorker::_c_grid_3d->_squares[current_sq_idx]->_yCenter, 
				                                                     -StuffWorker::_c_grid_3d->_squares[current_sq_idx]->_zCenter); // negative
			_c_pts_approx[_c_pts_approx_fill_size].second = 1;
			_c_pts_approx_fill_size++;
		}
	}
}


void AMass::Grow(float growth_scale_iter, float dt)
{
	// nothing happens
}

void AMass::Solve(const std::vector<A2DVector>& container, const AnElement& parentElem)
{
	if(_is_boundary)
	{
		if (_is_inside)
		{
			// ---------- OVERLAP FORCE ----------
			A3DVector sumO(0, 0, 0);
			A3DVector ctrPt;
			A3DVector dir;
			for (unsigned int a = 0; a < _triangles.size(); a++)
			{
				// todo maybe center of mass?
				ctrPt = (parentElem._massList[_triangles[a].idx0].GetPos() +        // triangle vertex
						 parentElem._massList[_triangles[a].idx1].GetPos() +        // triangle vertex
						 parentElem._massList[_triangles[a].idx2].GetPos()) / 3.0f; // triangle vertex

				dir = _pos.DirectionTo(ctrPt);
				sumO += dir;
			}
			sumO *= SystemParams::_k_overlap;
			 
			{ this->_overlapForce += sumO; }
		}
		else
		{
			// ---------- REPULSION FORCE ----------
			A3DVector sumR(0, 0, 0);
			A3DVector dir;
			float distSq;

			for (int a = 0; a < _c_pts_fill_size; a++)
			{
				dir = _c_pts[a].DirectionTo(_pos); // direction

				distSq = dir.LengthSquared(); // distance
				sumR += (dir.Norm() / (SystemParams::_repulsion_soft_factor + distSq));
			}


			for (int a = 0; a < _c_pts_approx_fill_size; a++)
			{
				dir = _c_pts_approx[a].first.DirectionTo(_pos); // direction
				distSq = dir.LengthSquared(); // distance
				sumR += (dir.Norm() *_c_pts_approx[a].second / (SystemParams::_repulsion_soft_factor + distSq));
			}

			sumR *= SystemParams::_k_repulsion;

			{
				this->_repulsionForce += sumR;
			}
		}
		
	}

	// ---------- BOUNDARY FORCE ----------
	float k_boundary = SystemParams::_k_boundary;
	if (!UtilityFunctions::InsidePolygon(container, _pos._x, _pos._y))
	{
		A2DVector pos2D = _pos.GetA2DVector();
		A2DVector cPt = UtilityFunctions::GetClosestPtOnClosedCurve(container, pos2D);
		A2DVector dirDist = pos2D.DirectionTo(cPt); // not normalized
		A2DVector bForce = dirDist * k_boundary;
		/*if (!bForce.IsBad())*/ 
		{ this->_boundaryForce += A3DVector(bForce.x, bForce.y, 0); } // z is always 0 !!!
	}

	// --------- DOCKING FORCE	
	if(_isDocked)
	{
		float k_dock = SystemParams::_k_dock;
		A3DVector dir = _pos.DirectionTo(_dockPoint); // not normalized
		float dist = dir.Length();
		dir = dir.Norm();
		A3DVector eForce = (dir * SystemParams::_k_dock * dist);
		/*if (!eForce.IsBad())*/
		{
			_edgeForce += eForce;	
		}
	}

	// --------- Z FORCE
	float k_z = SystemParams::_k_z;
	float z_dist = _ori_z_pos - _pos._z;
	_zForce = A3DVector(0, 0, z_dist) * k_z;

}