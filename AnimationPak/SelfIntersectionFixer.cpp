
/* ---------- ShapeRadiusMatching V2  ---------- */

#include "SelfIntersectionFixer.h"

#include "clipper.hpp"

// http:_//doc.cgal.org/latest/Sweep_line_2/index.html

#include <iostream>

// Computing intersection points among curves using the sweep line.
/*#include <CGAL/Cartesian.h>
#include <CGAL/MP_Float.h>
#include <CGAL/Quotient.h>
#include <CGAL/Arr_segment_traits_2.h>
#include <CGAL/Sweep_line_2_algorithms.h>
#include <list>

typedef CGAL::Quotient<CGAL::MP_Float>                  NT;
typedef CGAL::Cartesian<NT>                             Kernel;
typedef Kernel::Point_2                                 Point_2;
typedef CGAL::Arr_segment_traits_2<Kernel>              Traits_2;
typedef Traits_2::Curve_2                               Segment_2;*/

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <iostream>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point;
typedef CGAL::Polygon_2<K> Polygon_2;

SelfIntersectionFixer::SelfIntersectionFixer()
{
}

SelfIntersectionFixer::~SelfIntersectionFixer()
{
}

bool SelfIntersectionFixer::IsSimple(std::vector<A2DVector> poly)
{
	std::vector<Point> points;
	for (int a = 0; a < poly.size(); a++)
	{
		points.push_back(Point(poly[a].x, poly[a].y));
	}

	Polygon_2 pgn(points.begin(), points.end());
	// check if the polygon is simple.
	bool isSimple = pgn.is_simple();

	return isSimple;
}

void SelfIntersectionFixer::FixSelfIntersection1(std::vector<A2DVector> oldPoly, std::vector<A2DVector>& newPoly)
{
	float cScaling = 1e10; // because clipper only uses integer
	ClipperLib::Path oldCPoly;
	for (int a = 0; a < oldPoly.size(); a++)
	{
		oldCPoly << ClipperLib::IntPoint(oldPoly[a].x * cScaling, oldPoly[a].y * cScaling);
	}

	ClipperLib::Paths newCPolys;
	ClipperLib::SimplifyPolygon(oldCPoly, newCPolys);

	// find the largest
	float maxArea = std::numeric_limits<float>::min();
	ClipperLib::Path newCPoly;
	for (int a = 0; a < newCPolys.size(); a++)
	{
		float polyArea = std::abs(ClipperLib::Area(newCPolys[a]));
		if (polyArea > maxArea)
		{
			newCPoly = newCPolys[a];
			maxArea = polyArea;
		}
	}

	for (int b = 0; b < newCPoly.size(); b++)
	{
		A2DVector pt(newCPoly[b].X / cScaling, newCPoly[b].Y / cScaling);
		newPoly.push_back(pt);
	}
}
