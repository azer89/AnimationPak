
#include "TetWrapper.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/IO/Color.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel         K;
typedef CGAL::Triangulation_vertex_base_with_info_3<CGAL::Color, K> Vb;
typedef CGAL::Delaunay_triangulation_cell_base_3<K>                 Cb;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb>                Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds>                      Delaunay;
typedef Delaunay::Point                                             Point;

TetWrapper::TetWrapper()
{

}

TetWrapper::~TetWrapper()
{

}

void TetWrapper::GenerateTet(std::vector<A3DVector> input_points)
{
	Delaunay T;

	for (int a = 0; a < input_points.size(); a++)
	{
		T.insert(Point(input_points[a]._x, input_points[a]._y, input_points[a]._z));
	}

	Delaunay::Finite_vertices_iterator vit;
	for (vit = T.finite_vertices_begin(); vit != T.finite_vertices_end(); ++vit)
	{
		//if (T.degree(vit) == 6)
		//	vit->info() = CGAL::RED;
	}
		

}