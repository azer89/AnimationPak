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
struct A2DRectangle;


class UtilityFunctions
{
public:
	// array index
	static int GetIndexFromIntList(const std::vector<int>& aList, int elem);

	// rotation
	static A2DVector Rotate(A2DVector pt, A2DVector centerPt, float rad);
	static A2DVector Rotate(A2DVector pt, float rad);

	// in or out
	static bool InsidePolygon(const std::vector<A2DVector>& polygon, float px, float py);

	// closest
	static A2DVector GetClosestPtOnClosedCurve(const std::vector<A2DVector>& polyline, const A2DVector& p);
	static A2DVector ClosestPtAtFiniteLine2(const A2DVector& lnStart, const A2DVector& lnEnd, const A2DVector& pt);

	// distance
	static float DistanceToClosedCurve(std::vector<A2DVector> polyline, A2DVector p);
	static float DistanceToPolyline(const std::vector<A2DVector>& polyline, A2DVector p);
	static float DistanceToFiniteLine(A2DVector v, A2DVector w, A2DVector p);

	// resample
	static void UniformResample(std::vector<A2DVector> oriCurve, std::vector<A2DVector>& resampleCurve, float resampleGap);

	// curve
	static float CurveLengthClosed(std::vector<A2DVector> curves);
	static float CurveLength(std::vector<A2DVector> curves);

	//
	static A2DRectangle GetBoundingBox(std::vector<A2DVector> boundary);

	// translation
	static std::vector<A2DVector> TranslatePoly(std::vector<A2DVector> poly, float x, float y);
	static std::vector<A2DVector> MovePoly(std::vector<A2DVector> poly, A2DVector oldCenter, A2DVector newCenter);
};

#endif