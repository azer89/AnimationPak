
/* ---------- ShapeRadiusMatching V2  ---------- */

/*
================================================================================
Reza Adhitya Saputra
radhitya@uwaterloo.ca
August 2016
================================================================================
*/

#ifndef FLANN_PROXY_H
#define FLANN_PROXY_H

/*
this is a proxy of nanoflann library that has BSD license,
nanoflann builds a kd-tree which is really useful for KNN.

for more information about the library go to this github repo:
https://github.com/jlblancoc/nanoflann
*/

#include "nanoflann.hpp"
#include "A2DVector.h"
//#include "ALine.h"
#include "PointCloud2D.h"

#include <vector>

using namespace nanoflann;

// 
typedef KDTreeSingleIndexAdaptor< L2_Simple_Adaptor<float, PointCloud2D<float> >,
	PointCloud2D<float>,
	2 /*dim*/>
	PointKDTree;

class NANOFLANNWrapper2D
{
public:
	NANOFLANNWrapper2D();
	~NANOFLANNWrapper2D();

	// set data
	void SetPointData(std::vector<A2DVector> myData);
	void SetPointDataWithInfo(std::vector<A2DVector> myData, std::vector<int> info1, std::vector<int> info2);

	void AppendPointData(std::vector<A2DVector> myData);

	// query
	std::vector<A2DVector> GetClosestPoints(A2DVector pt, int num_query);
	std::vector<int>     GetClosestIndices(A2DVector pt, int num_query);

	std::vector<std::pair<int, int>> GetClosestPairIndices(A2DVector pt, int num_query);

	// prepare kd-tree
	void CreatePointKDTree();
	void CreatePointWithInfoKDTree();


public:
	//std::vector<std::vector<AVector>> _djikstraData;

	std::vector<A2DVector> _pointData;
	std::vector<int> _pointInfo1;
	std::vector<int> _pointInfo2;

	//std::vector<ALine>   _lineData;

	PointKDTree* 	  _pointKDTree;
	PointCloud2D<float> _pointCloud;

	int _leaf_max_size;
};

#endif