
/* ---------- AnimationPak  ---------- */

#include "UtilityFUnctions.h"

#include "A2DVector.h"
#include "A3DVector.h"
//#include "ALine.h"
#include "A2DRectangle.h"
#include "ABary.h"

#include <sstream>

#include <algorithm>

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
std::vector<A2DVector>  UtilityFunctions::Convert2Dto3D(std::vector<A3DVector> poly)
{
	std::vector<A2DVector> twoDArray;
	for (unsigned int a = 0; a < poly.size(); a++)
	{
		twoDArray.push_back(A2DVector(poly[a]._x, poly[a]._y));
	}
	return twoDArray;
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

/*
================================================================================
================================================================================
*/
float UtilityFunctions::DistanceToBunchOfPoints(const std::vector<A3DVector>& points, A3DVector p)
{
	float dist = std::numeric_limits<float>::max();
	for (unsigned int b = 0; b < points.size(); b++)
	{
		float d = p.DistanceSquared(points[b]);
		if (d < dist) { dist = d; }
	}
	return std::sqrt(dist);
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


/*================================================================================
================================================================================*/

float clip(float n, float lower, float upper) {
	return std::max(lower, std::min(n, upper));
}

A3DVector UtilityFunctions::ClosestPointOnTriangle2(A3DVector p, A3DVector a, A3DVector b, A3DVector c)
{
	// Check if P in vertex region outside A
	A3DVector ab = b - a;
	A3DVector ac = c - a;
	A3DVector ap = p - a;
	float d1 = ab.Dot(ap);
	float d2 = ac.Dot(ap);
	if (d1 <= 0.0f && d2 <= 0.0f) return a; // barycentric coordinates (1, 0, 0)

	// Check if P in vertex region outside B
	A3DVector bp = p - b;
	float d3 = ab.Dot(bp);
	float d4 = ac.Dot(bp);
	if (d3 >= 0.0f && d4 <= d3) return b; // barycentric coordinates (0, 1, 0)

	// Check if P in edge region of AB, if so return projection of P onto AB
	float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
	{
		float v = d1 / (d1 - d3);
		return a + ab * v; // barycentric coordinates (1-v, v, 0)
	}

	// Check if P in vertex region outside C
	A3DVector cp = p - c;
	float d5 = ab.Dot(cp);
	float d6 = ac.Dot(cp);
	if (d6 >= 0.0f && d5 <= d6) return c; // barycentric coordinates (0, 0, 1)

	// Check if P in edge region of AC, if so return projection of P onto AC
	float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
	{
		float w = d2 / (d2 - d6);
		return a + ac * w; // barycentric coordinates (1-w, 0, w)
	}

	// Check if P in edge region of BC, if so return projection of P onto BC
	float va = d3 * d6 - d5 * d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
	{
		float w = (d4 - d3) / ( (d4 - d3) + (d5 - d6) );
		return b + (c - b) * w; // barycentric coordinates (0, 1-w, w)
	}

	// P inside face region. Computer Q through its barycentric coordinates (u, v, w)
	float denom = 1.0f / (va + vb + vc);
	float v = vb * denom;
	float w = vc * denom;
	return a + ab * v + ac * w; 

}

A3DVector UtilityFunctions::ClosestPointOnTriangle(std::vector<A3DVector>& triangle, A3DVector sourcePosition)
{
	A3DVector edge0 = triangle[1] - triangle[0];
	A3DVector edge1 = triangle[2] - triangle[0];
	A3DVector v0 = triangle[0] - sourcePosition;

	float a = edge0.Dot(edge0);
	float b = edge0.Dot(edge1);
	float c = edge1.Dot(edge1);
	float d = edge0.Dot(v0);
	float e = edge1.Dot(v0);

	float det = a * c - b * b;
	float s = b * e - c * d;
	float t = b * d - a * e;

	if (s + t < det)
	{
		if (s < 0.f)
		{
			if (t < 0.f)
			{
				if (d < 0.f)
				{
					s = clip(-d / a, 0.f, 1.f);
					t = 0.f;
				}
				else
				{
					s = 0.f;
					t = clip(-e / c, 0.f, 1.f);
				}
			}
			else
			{
				s = 0.f;
				t = clip(-e / c, 0.f, 1.f);
			}
		}
		else if (t < 0.f)
		{
			s = clip(-d / a, 0.f, 1.f);
			t = 0.f;
		}
		else
		{
			float invDet = 1.f / det;
			s *= invDet;
			t *= invDet;
		}
	}
	else
	{
		if (s < 0.f)
		{
			float tmp0 = b + d;
			float tmp1 = c + e;
			if (tmp1 > tmp0)
			{
				float numer = tmp1 - tmp0;
				float denom = a - 2 * b + c;
				s = clip(numer / denom, 0.f, 1.f);
				t = 1 - s;
			}
			else
			{
				t = clip(-e / c, 0.f, 1.f);
				s = 0.f;
			}
		}
		else if (t < 0.f)
		{
			if (a + d > b + e)
			{
				float numer = c + e - b - d;
				float denom = a - 2 * b + c;
				s = clip(numer / denom, 0.f, 1.f);
				t = 1 - s;
			}
			else
			{
				s = clip(-e / c, 0.f, 1.f);
				t = 0.f;
			}
		}
		else
		{
			float numer = c + e - b - d;
			float denom = a - 2 * b + c;
			s = clip(numer / denom, 0.f, 1.f);
			t = 1.f - s;
		}
	}

	return triangle[0] + (edge0 * s) + (edge1 * t);
}

/*
================================================================================
================================================================================
*/
bool UtilityFunctions::HasEnding(std::string const &fullString, std::string const &ending)
{
	if (fullString.length() >= ending.length())
	{
		return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
	}
	else
	{
		return false;
	}
}

/*
================================================================================
================================================================================
*/
// split string
std::vector<std::string>& UtilityFunctions::Split(const std::string &s, char delim, std::vector<std::string> &elems)
{
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim))
	{
		elems.push_back(item);
	}
	return elems;
}

/*
================================================================================
================================================================================
*/
// split string
std::vector<std::string> UtilityFunctions::Split(const std::string &s, char delim)
{
	std::vector<std::string> elems;
	UtilityFunctions::Split(s, delim, elems);
	return elems;
}

/*
================================================================================
================================================================================
*/
ABary UtilityFunctions::Barycentric(A2DVector p, A2DVector A, A2DVector B, A2DVector C)
{
	ABary bary;

	A2DVector v0 = B - A;
	A2DVector v1 = C - A;
	A2DVector v2 = p - A;
	float d00 = v0.Dot(v0);
	float d01 = v0.Dot(v1);
	float d11 = v1.Dot(v1);
	float d20 = v2.Dot(v0);
	float d21 = v2.Dot(v1);
	float denom = d00 * d11 - d01 * d01;
	bary._v = (d11 * d20 - d01 * d21) / denom;
	bary._w = (d00 * d21 - d01 * d20) / denom;
	bary._u = 1.0 - bary._v - bary._w;

	//if (bary._v < 0 || bary._v > 1.0) { std::cout << "bary._v : " << bary._v << "\n"; }
	//if (bary._w < 0 || bary._w > 1.0) { std::cout << "bary._w : " << bary._w << "\n"; }
	//if (bary._u < 0 || bary._u > 1.0) { std::cout << "bary._u : " << bary._u << "\n"; }

	return bary;
}

/*
================================================================================
Return the angle between two vectors on a plane
The angle is from vector 1 to vector 2, positive anticlockwise
The result is between -pi -> pi
================================================================================
*/
float UtilityFunctions::Angle2D(float x1, float y1, float x2, float y2)
{
	// atan2(vector.y, vector.x) = the angle between the vector and the X axis

	float dtheta, theta1, theta2;

	theta1 = atan2(y1, x1);
	theta2 = atan2(y2, x2);
	dtheta = theta2 - theta1;

	while (dtheta > PI)
	{
		dtheta -= PI2;
	}

	while (dtheta < -PI)
	{
		dtheta += PI2;
	}

	return dtheta;
}

std::vector < std::vector<A2DVector>> UtilityFunctions::FlipY(std::vector < std::vector<A2DVector>> polys, float yCenter)
{
	std::vector < std::vector<A2DVector>> flipped_polys;
	for (int a = 0; a < polys.size(); a++)
	{
		flipped_polys.push_back(UtilityFunctions::FlipY(polys[a], yCenter));
	}
	return flipped_polys;
}

std::vector<A2DVector> UtilityFunctions::FlipY(std::vector<A2DVector> poly, float yCenter)
{
	std::vector<A2DVector> flipped_poly(poly.size());

	for (unsigned int a = 0; a < poly.size(); a++)
	{
		float y_delta = yCenter - poly[a].y;
		flipped_poly[a].x = poly[a].x;
		flipped_poly[a].y = yCenter + y_delta;
	}

	return flipped_poly;
}

A2DVector UtilityFunctions::FlipY(A2DVector pt, float yCenter)
{
	float y_delta = yCenter - pt.y;

	return A2DVector(pt.x, yCenter + y_delta);
}