
#include "TetWrapper.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/IO/Color.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel         K;
typedef CGAL::Triangulation_vertex_base_with_info_3<int, K> Vb;
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

bool TetWrapper::ValidIndex(int idx)
{
	if (idx < 0) return false;
	if (idx >= _massSize) return false;
	return true;
}

//void TetWrapper::PruneEdges(const std::vector<AMass>& massList,
//	                        const std::vector<AnIndexedLine>& tetEdges)
//{
//	for (int a = 0; a < tetEdges.size(); a++)
//	{
//		AnIndexedLine ln = tetEdges[a];
//		int layer_idx1 = massList[ln._index0]._layer_idx;
//		int layer_idx2 = massList[ln._index1]._layer_idx;
//	}
//}

void TetWrapper::GenerateTet(const std::vector<AMass>& massList, float maxDistRandPt, std::vector<AnIndexedLine>& tetEdges)
{
	_massSize = massList.size();

	Delaunay T;

	for (int a = 0; a < massList.size(); a++)
	{
		T.insert(Point(massList[a]._pos._x, massList[a]._pos._y, massList[a]._pos._z));
	}

	Delaunay::Finite_vertices_iterator vit;
	int iter = 0;
	for (vit = T.finite_vertices_begin(); vit != T.finite_vertices_end(); ++vit)
	{
		//if (T.degree(vit) == 6)
		vit->info() = iter;
		iter++;
	}

	//std::vector<AnIndexedLine> tetEdges;

	// iterate edge
	for (Delaunay::Finite_edges_iterator eit = T.finite_edges_begin();
		eit != T.finite_edges_end();
		eit++)
	{
		int idx1 = eit->first->vertex(eit->second)->info();
		int idx2 = eit->first->vertex(eit->third)->info();

		float zGap = ((float)SystemParams::_upscaleFactor) / ((float)(SystemParams::_num_layer - 1));
		float maxLen1 = sqrt(zGap * zGap + maxDistRandPt * maxDistRandPt) + 0.01;
		float maxLen2 = sqrt( 2.0 * (maxDistRandPt * maxDistRandPt) ) * 1.2;
		//float maxLen2 = maxDistRandPt;

		if(ValidIndex(idx1) && ValidIndex(idx2))
		{
			A3DVector pt1 = massList[idx1]._pos;
			A3DVector pt2 = massList[idx2]._pos;

			float maxLen = maxLen1;
			if (massList[idx1]._layer_idx == massList[idx2]._layer_idx) { maxLen = maxLen2; }

			if(pt1.Distance(pt2) < maxLen)
			{
				tetEdges.push_back(AnIndexedLine(idx1, idx2));
			}
			//std::cout << idx1 << ", " << idx2 << "\n";
		}
	}
		

	
	//std::cout << "done" << "\n";
	//return tetEdges;
}