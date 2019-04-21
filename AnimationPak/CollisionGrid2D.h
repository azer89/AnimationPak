
#ifndef __Collision_Grid_2D__
#define __Collision_Grid_2D__

#include <vector>

#include "SystemParams.h"
#include "A2DSquare.h"
#include "A2DVector.h"

#include <cmath>

// to read:

// http_://gamedev.stackexchange.com/questions/72030/using-uniform-grids-for-collision-detection-efficient-way-to-keep-track-of-wha

typedef std::vector<int> GraphIndices;

class CollisionGrid2D
{
public:
	CollisionGrid2D();

	CollisionGrid2D(float cellSize);

	~CollisionGrid2D();

	void GetCellPosition(int& xPos, int& yPos, float x, float y);

	//void InsertAPoint(float x, float y, int info1, int info2, float nx, float ny);

	void InsertAPoint(float x, float y, int info1, int info2);

	void GetGraphIndices1(float x, float y, std::vector<int>& closestGraphIndices);

	void GetGraphIndices1B(float x, float y, std::vector<int>& closestGraphIndices);

	void GetGraphIndices2(float x, float y, int parentGraphIndex, std::vector<int>& closestGraphIndices);

	void GetGraphIndices2B(float x, float y, std::vector<int>& closestGraphIndices);

	void GetData(float x, float y, int parentGraphIndex, std::vector<A2DVector>& closestPts, std::vector<int>& closestGraphIndices);

	void GetClosestPoints(float x, float y, std::vector<A2DVector>& closestPts);

	std::vector<A2DObject*> GetObjects(float x, float y);

	void MovePoints();

	//void Draw();

	//void AnalyzeContainer(const std::vector<std::vector<AVector>>&  boundaries,
	//	const std::vector<std::vector<AVector>>&  holes,
	//	const std::vector<std::vector<AVector>>& offsetFocalBoundaries);

	void PrecomputeGraphIndices();

	//bool NearBoundary(float x, float y);

public:
	std::vector<GraphIndices> _graphIndexArray;

	int _numColumn;
	float _maxLength;

	std::vector<A2DObject*> _objects;

	std::vector<A2DSquare*> _squares;
};

#endif