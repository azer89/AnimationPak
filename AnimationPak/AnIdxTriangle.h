
// ---------- ShapeRadiusMatching V2 ----------
/**
*
* Triangles with three vertices
* Each vertices doesn't hold the real coordinate
*
* Author: Reza Adhitya Saputra (reza.adhitya.saputra@gmail.com)
* Version: 2019
*
*
*/

#ifndef __An_Idx_Triangle_h__
#define __An_Idx_Triangle_h__

#include "A2DVector.h"

#include "A3DVector.h"

struct AnIdxTriangle
{
public:
	// first index
	int idx0;

	// second index
	int idx1;

	// third index
	int idx2;

	int _layer_idx;

	A3DVector _temp_1_3d; // temporary
	A3DVector _temp_2_3d; // temporary
	A3DVector _temp_3_3d; // temporary
	A3DVector _temp_center_3d; // temporary

	// type of the triangle (black, white, and screentone)
	//TriangleType tri_type;

	// Constructor
	AnIdxTriangle(int idx0, int idx1, int idx2)
	{
		this->idx0 = idx0;
		this->idx1 = idx1;
		this->idx2 = idx2;

		this->_layer_idx = -1;
	}

	AnIdxTriangle(int idx0, int idx1, int idx2, int layer_idx)
	{
		this->idx0 = idx0;
		this->idx1 = idx1;
		this->idx2 = idx2;

		this->_layer_idx = layer_idx;
	}
};
//}

#endif