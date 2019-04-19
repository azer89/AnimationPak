
/* ---------- AnimationPak  ---------- */

#include "UtilityFUnctions.h"

#include "A2DVector.h"
//#include "ALine.h"
#include "A2DRectangle.h"

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

/*
================================================================================
================================================================================
*/
float UtilityFunctions::CurveLengthClosed(std::vector<A2DVector> curves)
{
	curves.push_back(curves[curves.size() - 1]);
	return CurveLength(curves);
}

/*
================================================================================
================================================================================
*/
float UtilityFunctions::CurveLength(std::vector<A2DVector> curves)
{
	float length = 0.0;
	for (size_t a = 1; a < curves.size(); a++) { length += curves[a].Distance(curves[a - 1]); }
	return length;
}

/*
================================================================================
================================================================================
*/
void UtilityFunctions::UniformResample(std::vector<A2DVector> oriCurve, std::vector<A2DVector>& resampleCurve, float resampleGap)
{
	resampleCurve.clear();
	float curveLength = CurveLength(oriCurve);

	int segmentNum = (int)(std::round(curveLength / resampleGap)); // rounding
	resampleGap = curveLength / (float)segmentNum;

	int iter = 0;
	float sumDist = 0.0;
	resampleCurve.push_back(oriCurve[0]);
	while (iter < oriCurve.size() - 1)
	{
		float currentDist = oriCurve[iter].Distance(oriCurve[iter + 1]);
		sumDist += currentDist;

		if (sumDist > resampleGap)
		{
			float vectorLength = currentDist - (sumDist - resampleGap);
			A2DVector pt1 = oriCurve[iter];
			A2DVector pt2 = oriCurve[iter + 1];
			A2DVector directionVector = (pt2 - pt1).Norm();

			A2DVector newPoint1 = pt1 + directionVector * vectorLength;
			resampleCurve.push_back(newPoint1);

			sumDist = currentDist - vectorLength;

			while (sumDist - resampleGap > 1e-8)
			{
				A2DVector insertPt2 = resampleCurve[resampleCurve.size() - 1] + directionVector * resampleGap;
				resampleCurve.push_back(insertPt2);
				sumDist -= resampleGap;
			}
		}

		iter++;

	}



	float eps = std::numeric_limits<float>::epsilon();
	A2DVector lastPt = oriCurve[oriCurve.size() - 1];
	if (resampleCurve[resampleCurve.size() - 1].Distance(lastPt) > (resampleGap - eps)) { resampleCurve.push_back(lastPt); }

}

/*
================================================================================
================================================================================
*/
A2DRectangle UtilityFunctions::GetBoundingBox(std::vector<A2DVector> boundary)
{
	/*std::vector<cv::Point2f> newBoundary;
	for (int a = 0; a < boundary.size(); a++)
	{ newBoundary.push_back(cv::Point2f(boundary[a].x, boundary[a].y)); }
	cv::Rect bb = cv::boundingRect(newBoundary);
	return ARectangle(AVector(bb.x, bb.y), bb.width, bb.height);*/

	float xMax = std::numeric_limits<float>::min();
	float yMax = std::numeric_limits<float>::min();
	float xMin = std::numeric_limits<float>::max();
	float yMin = std::numeric_limits<float>::max();

	for (unsigned int a = 0; a < boundary.size(); a++)
	{
		A2DVector pt = boundary[a];

		if (pt.x > xMax) { xMax = pt.x; }
		if (pt.y > yMax) { yMax = pt.y; }
		if (pt.x < xMin) { xMin = pt.x; }
		if (pt.y < yMin) { yMin = pt.y; }
	}

	return A2DRectangle(A2DVector(xMin, yMin), xMax - xMin, yMax - yMin);
}

/*
================================================================================
================================================================================
*/
std::vector<A2DVector> UtilityFunctions::TranslatePoly(std::vector<A2DVector> poly, float x, float y)
{
	std::vector<A2DVector> newPoly;
	for (unsigned int a = 0; a < poly.size(); a++)
	{
		newPoly.push_back(poly[a] + A2DVector(x, y));
	}
	return newPoly;
}

/*
================================================================================
================================================================================
*/
std::vector<A2DVector> UtilityFunctions::MovePoly(std::vector<A2DVector> poly, A2DVector oldCenter, A2DVector newCenter)
{
	A2DVector offsetVector = newCenter - oldCenter;
	return TranslatePoly(poly, offsetVector.x, offsetVector.y);
}



/*
================================================================================
================================================================================
*/
float UtilityFunctions::DistanceToClosedCurve(std::vector<A2DVector> polyline, A2DVector p)
{
	polyline.push_back(polyline[0]); // because loop
	return UtilityFunctions::DistanceToPolyline(polyline, p);
}

/*
================================================================================
================================================================================
*/
float UtilityFunctions::DistanceToPolyline(const std::vector<A2DVector>& polyline, A2DVector p)
{
	float dist = std::numeric_limits<float>::max();
	for (unsigned int a = 1; a < polyline.size(); a++)
	{
		float d = DistanceToFiniteLine(polyline[a - 1], polyline[a], p);
		if (d < dist) { dist = d; }
	}
	return dist;
}

/*================================================================================
================================================================================*/
float UtilityFunctions::DistanceToFiniteLine(A2DVector v, A2DVector w, A2DVector p)
{
	float machine_eps = std::numeric_limits<float>::epsilon();

	// Return minimum distance between line segment vw and point p
	float l2 = v.DistanceSquared(w);					   // i.e. |w-v|^2 -  avoid a sqrt
	if (l2 > -machine_eps && l2 < machine_eps) return p.Distance(v);   // v == w case

																	   // Consider the line extending the segment, parameterized as v + t (w - v).
																	   // We find projection of point p onto the line. 
																	   // It falls where t = [(p-v) . (w-v)] / |w-v|^2
	float t = (p - v).Dot(w - v) / l2;

	if (t < 0.0) { return  p.Distance(v); }  // Beyond the 'v' end of the segment
	else if (t > 1.0) { return  p.Distance(w); }  // Beyond the 'w' end of the segment
	A2DVector projection = v + (w - v) * t;         // Projection falls on the segment
	return p.Distance(projection);
}