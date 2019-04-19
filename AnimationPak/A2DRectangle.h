
/* ---------- AnimationPak  ---------- */

#ifndef A2DRECTANGLE_H
#define A2DRECTANGLE_H

#include "A2DVector.h"

/**
* Reza Adhitya Saputra
* radhitya@uwaterloo.ca
* June 2016
*/

struct A2DRectangle
{
public:
	A2DVector topleft;
	float witdh;
	float height;

	A2DRectangle()
	{

	}

	A2DRectangle(A2DVector topleft, float witdh, float height)
	{
		this->topleft = topleft;
		this->witdh = witdh;
		this->height = height;
	}

	A2DVector GetCenter()
	{
		return A2DVector(topleft.x + witdh / 2.0f, topleft.y + height / 2.0f);
	}
};



#endif 