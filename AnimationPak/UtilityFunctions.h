/* ---------- AnimationPak  ---------- */

/*
================================================================================
Reza Adhitya Saputra
radhitya@uwaterloo.ca
================================================================================
*/

#ifndef UTILITY_FUNCTIONS_H
#define UTILITY_FUNCTIONS_H

#include <vector>
//#include "APath.h"
//#include "ABary.h"

// forward declaration
struct A2DVector;
struct ALine;
struct ARectangle;


class UtilityFunctions
{
public:
	static int GetIndexFromIntList(const std::vector<int>& aList, int elem);

	static A2DVector Rotate(A2DVector pt, A2DVector centerPt, float rad);
	static A2DVector Rotate(A2DVector pt, float rad);

	static bool InsidePolygon(const std::vector<A2DVector>& polygon, float px, float py);

	static A2DVector GetClosestPtOnClosedCurve(const std::vector<A2DVector>& polyline, const A2DVector& p);
	static A2DVector ClosestPtAtFiniteLine2(const A2DVector& lnStart, const A2DVector& lnEnd, const A2DVector& pt);
};

#endif