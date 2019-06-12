
#ifndef __Collision_Grid_3D__
#define __Collision_Grid_3D__

#include <vector>

#include "SystemParams.h"
#include "A3DSquare.h"
#include "A3DVector.h"

#include <cmath>

typedef std::vector<int> GraphIndices;
typedef std::vector<std::vector<int>> TriangleIndices;

class CollisionGrid3D
{
private:
	std::vector<A3DObject*>   _objects;

public:
	
	std::vector<A3DSquare*>   _squares;

	std::vector<GraphIndices> _graphIndexArray; // per squares
	std::vector<TriangleIndices> _triangleIndexArray; // per squares

	std::vector<GraphIndices> _approx_graphIndexArray; // per squares
	std::vector<TriangleIndices> _approx_triangleIndexArray; // per squares

	int   _side_num; // side length of the entire grid (unit: cell)
	float _max_cell_length; // side length of a cell

public:
	CollisionGrid3D();

	//CollisionGrid3D(float cellSize);
	void Init();

	void SetPoint(int idx, A3DVector pos);

	~CollisionGrid3D();	

	void InsertAPoint(float x, float y, float z, int info1, int info2); // make z positive

	void GetClosestPoints(float x, float y, float z, std::vector<A3DVector>& closestPoints);

	void GetClosestObjects(float x, float y, float z, std::vector<A3DObject>& closestObjects);

	void GetGraphIndices2B(float x, float y, float z, std::vector<int>& closestGraphIndices);

	void GetTriangleIndices(float x, float y, float z, TriangleIndices& closestTriIndices);

	void PrecomputeClosestGraphsAndTriangles2();

	void PrecomputeClosestGraphsAndTriangles();

	void MovePoints();

private:
	void GetCellPosition(int& xPos, int& yPos, int& zPos, float x, float y, float z);

	int SquareIndex(int xPos, int yPos, int zPos);

};

#endif
