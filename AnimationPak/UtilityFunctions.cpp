
/* ---------- AnimationPak  ---------- */

#include "UtilityFUnctions.h"

#include "A2DVector.h"
//#include "ALine.h"
//#include "ARectangle.h"

#include <sstream>

#define PI 3.14159265359
#define PI2 6.28318530718

/*================================================================================
paulbourke.net/geometry/pointlineplane/
================================================================================*/
inline float DistSquared(const A2DVector& p, const A2DVector& other)
{
	float xDist = p.x - other.x;
	float yDist = p.y - other.y;
	return xDist * xDist + yDist * yDist;
}

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

/*
================================================================================
Must be counter-clockwise (0,0 is at topleft)
================================================================================
*/
bool UtilityFunctions::InsidePolygon(const std::vector<A2DVector>& polygon, float px, float py)
{
	// http_:_//alienryderflex_._com/polygon/

	int poly_sz = polygon.size();
	unsigned int   i, j = poly_sz - 1;
	bool  oddNodes = false;


	for (i = 0; i < poly_sz; i++)
	{
		if ((polygon[i].y < py && polygon[j].y >= py ||
			polygon[j].y < py && polygon[i].y >= py)
			&& (polygon[i].x <= px || polygon[j].x <= px))
		{
			oddNodes ^= (polygon[i].x + (py - polygon[i].y) / (polygon[j].y - polygon[i].y) * (polygon[j].x - polygon[i].x) < px);
		}
		j = i;
	}

	return oddNodes;
}

/*
================================================================================
================================================================================
*/
A2DVector UtilityFunctions::GetClosestPtOnClosedCurve(const std::vector<A2DVector>& polyline, const A2DVector& p)
{
	float dist = 10000000000;
	A2DVector closestPt;
	A2DVector pt;
	float d;
	int p_size = polyline.size();
	for (unsigned int a = 1; a < p_size; a++)
	{
		pt = ClosestPtAtFiniteLine2(polyline[a - 1], polyline[a], p);
		d = DistSquared(p, pt); // p.DistanceSquared(pt);
		if (d < dist)
		{
			dist = d;
			closestPt = pt;
		}
	}
	{ // first and last point
		pt = ClosestPtAtFiniteLine2(polyline[p_size - 1], polyline[0], p);
		d = DistSquared(p, pt); //p.DistanceSquared(pt);
		if (d < dist)
		{
			dist = d;
			closestPt = pt;
		}
	}
	return closestPt;
}

/*
================================================================================
================================================================================
*/
A2DVector UtilityFunctions::ClosestPtAtFiniteLine2(const A2DVector& lnStart, const A2DVector& lnEnd, const A2DVector& pt)
{
	float dx = lnEnd.x - lnStart.x;
	float dy = lnEnd.y - lnStart.y;

	float lineMagSq = DistSquared(lnStart, lnEnd); //lnStart.DistanceSquared(lnEnd);

	float u = (((pt.x - lnStart.x) * dx) +
		((pt.y - lnStart.y) * dy)) /
		lineMagSq;

	if (u < 0.0f) { return lnStart; }
	else if (u > 1.0f) { return lnEnd; }

	return A2DVector(lnStart.x + u * dx, lnStart.y + u * dy);
}


