
/* ---------- AnimationPak  ---------- */

#include "UtilityFUnctions.h"

#include "A2DVector.h"
//#include "ALine.h"
//#include "ARectangle.h"

#include <sstream>

#define PI 3.14159265359
#define PI2 6.28318530718

/*
================================================================================
================================================================================
*/
int UtilityFunctions::GetIndexFromIntList(const std::vector<int>& aList, int elem)
{
	for (unsigned int a = 0; a < aList.size(); a++)
	{
		if (elem == aList[a]) { return a; }
	}

	return -1;
}

/*
================================================================================
================================================================================
*/
A2DVector UtilityFunctions::Rotate(A2DVector pt, A2DVector centerPt, float rad)
{
	pt -= centerPt;
	pt = UtilityFunctions::Rotate(pt, rad);
	pt += centerPt;
	return pt;
}

/*
================================================================================
================================================================================
*/
A2DVector UtilityFunctions::Rotate(A2DVector pt, float rad)
{
	float cs = cos(rad);
	float sn = sin(rad);

	float x = pt.x * cs - pt.y * sn;
	float y = pt.x * sn + pt.y * cs;

	return A2DVector(x, y);
}
