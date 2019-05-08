
#include "AMass.h"

#include "SystemParams.h"

#include "StuffWorker.h"

#include "UtilityFunctions.h"

AMass::AMass()
{
	//this->_m     = 0;             // mass is always one
	this->_pos = A3DVector(0, 0, 0);
	this->_self_idx = -1;
	//this->_cellIdx = -1;

	CallMeFromConstructor();
}

// Constructor
AMass::AMass(float x, float y, float z)
{
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
	this->_pos = pos;
	this->_self_idx = -1;

	CallMeFromConstructor();
}

AMass::~AMass()
{
}

void AMass::CallMeFromConstructor()
{
	_velocity = A3DVector(0, 0, 0);

	// hard parameter for closest pt search
	_closestPt_fill_sz = 0;
	_closestPt_actual_sz = 150; // BE CAREFUL HARD CODED PARAM!!!
	_closestPoints = std::vector <A2DVector>(_closestPt_actual_sz);

	_is_inside = false;

	_isDocked = false;

	//_interpolation_mode = false;

	

	Init();
}

void AMass::Init()
{
	//_attractionForce = AVector(0, 0);
	this->_edgeForce      = A3DVector(0, 0, 0);
	this->_repulsionForce = A3DVector(0, 0, 0);
	this->_boundaryForce  = A3DVector(0, 0, 0);
	this->_overlapForce   = A3DVector(0, 0, 0);
	this->_rotationForce  = A3DVector(0, 0, 0);
}

void AMass::Simulate(float dt)
{
	//if (_isDocked) { return; }

	// oiler
	_velocity += ((_edgeForce +
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



void AMass::GetClosestPoint()
{
	//std::cout << "closestpt";
	if (!_is_boundary) { return; }
	if (_parent_idx < 0 || _parent_idx >= StuffWorker::_element_list.size()) { return; }
	//if (this->_idx >= StuffWorker::_graphs[parentGraphIndex]._skinPointNum) { return; } // uncomment me

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

			/*if (UtilityFunctions::InsidePolygon(StuffWorker::_graphs[_closestGraphIndices[a]]._skin, _pos.x, _pos.y))*/
			if (UtilityFunctions::InsidePolygon(StuffWorker::_element_list[_closestGraphIndices[a]]._per_layer_boundary[_layer_idx], _pos._x, _pos._y))
			{
				insideGraphFlags.push_back(true);
				_is_inside = true;
				std::cout << "is inside\n";
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

			//A2DVector pt = UtilityFunctions::GetClosestPtOnClosedCurve(StuffWorker::_element_list[_closestGraphIndices[a]]._skin, _pos);
			A2DVector pt = StuffWorker::_element_list[_closestGraphIndices[a]].ClosestPtOnALayer(_pos.GetA2DVector(), _layer_idx);
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
}

void AMass::Grow(float growth_scale_iter, float dt)
{
	// nothing happens
}

void AMass::Solve(const std::vector<A2DVector>& container)
{
	if (_self_idx % StuffWorker::_element_list[_parent_idx]._numPointPerLayer <  StuffWorker::_element_list[_parent_idx]._numBoundaryPointPerLayer)
	{
		if (_closestGraphIndices.size() > 0)
		{
			if (_is_inside)
			{
				std::cout << "overlap force\n";
				// ---------- OVERLAP FORCE ----------
				A2DVector sumO(0, 0);
				A2DVector ctrPt;
				A2DVector dir;
				for (unsigned int a = 0; a < _triangles.size(); a++)
				{
					ctrPt = (StuffWorker::_element_list[_parent_idx]._massList[_triangles[a].idx0]._pos.GetA2DVector() +        // triangle vertex
						     StuffWorker::_element_list[_parent_idx]._massList[_triangles[a].idx1]._pos.GetA2DVector() +        // triangle vertex
						     StuffWorker::_element_list[_parent_idx]._massList[_triangles[a].idx2]._pos.GetA2DVector()) / 3.0f; // triangle vertex

					dir = _pos.GetA2DVector().DirectionTo(ctrPt);
					sumO += dir;
				}
				sumO *= SystemParams::_k_overlap;
				if (!sumO.IsBad()) { this->_overlapForce += A3DVector(sumO.x, sumO.y, 0); }
			}
			else
			{
				// ---------- REPULSION FORCE ----------
				//std::cout << _closestPt_fill_sz << " ";
				A2DVector sumR(0, 0);
				A2DVector dir;
				for (int a = 0; a < _closestPt_fill_sz; a++)
				{
					dir = _closestPoints[a].DirectionTo(_pos.GetA2DVector()); // direction
					float dist = dir.Length(); // distance
					sumR += (dir.Norm() / (SystemParams::_repulsion_soft_factor + std::pow(dist, 2)));
				}
				sumR *= SystemParams::_k_repulsion;
				if (!sumR.IsBad())
				{
					this->_repulsionForce += A3DVector(sumR.x, sumR.y, 0); // z is always 0 !!!
				}
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
		if (!bForce.IsBad()) { this->_boundaryForce += A3DVector(bForce.x, bForce.y, 0); } // z is always 0 !!!
	}

	// --------- DOCKING FORCE	
	if(_isDocked)
	{
		float k_dock = SystemParams::_k_dock;
		A3DVector dir = _pos.DirectionTo(_dockPoint); // not normalized
		float dist = dir.Length();
		dir = dir.Norm();
		A3DVector eForce = (dir * SystemParams::_k_dock * dist);
		if (!eForce.IsBad())
		{
			_edgeForce += eForce;	
		}
	}
}

/*void AMass::EnableInterpolationMode()
{

}*/

/*void AMass::DisableInterpolationMode()
{

}*/

/*void AMass::GetClosestPoint()
{
	if (_parent_idx < 0 || _parent_idx >= StuffWorker::_element_list.size()) { return; }
	//if (this->_idx >= StuffWorker::_graphs[parentGraphIndex]._skinPointNum) { return; } // uncomment me
	
	this->_closestGraphIndices.clear();
	//this->_closestPoints.clear();
	this->_closestPt_fill_sz = 0;
	this->_is_inside = false;           // "inside" flag
	
	_c_grid->GetGraphIndices2B(_pos._x, _pos._y, _parent_idx, _closestGraphIndices);
	
	if (_closestGraphIndices.size() > 0)
	{
		std::vector<bool> insideGraphFlags;
		int sz = _closestGraphIndices.size();
		for (unsigned int a = 0; a < sz; a++)
		{
			// uncomment me
			if (_closestGraphIndices[a] == _parent_idx) { insideGraphFlags.push_back(true); continue; }

			if (UtilityFunctions::InsidePolygon(StuffWorker::_graphs[_closestGraphIndices[a]]._skin, _pos.x, _pos.y))
			{
				insideGraphFlags.push_back(true);
				_isInside = true;
				continue; // can be more than one
			}
			else
			{ insideGraphFlags.push_back(false); }
		}

		// closest pts
		int sz2 = sz;
		if (sz2 > _closestPt_actual_sz) { sz2 = _closestPt_actual_sz; }
		for (unsigned int a = 0; a < sz2; a++)
		{
			if (insideGraphFlags[a]) { continue; }

			A2DVector pt = UtilityFunctions::GetClosestPtOnClosedCurve(StuffWorker::_element_list[_closestGraphIndices[a]]._skin, _pos);
			_closestPoints[_closestPt_fill_sz] = pt;
			_closestPt_fill_sz++;
		}
	}

	// this is used in AGraph
	_closestDist = std::numeric_limits<float>::max();
	for (unsigned int a = 0; a < _closestPt_fill_sz; a++)
	{
		float d = _closestPoints[a].DistanceSquared(_pos); // SQUARED!!!
		if (d < _closestDist)
		{
			_closestDist = d;
		}
	}
	_closestDist = std::sqrt(_closestDist); // SQRT
}*/
