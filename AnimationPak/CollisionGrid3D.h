
#ifndef __Collision_Grid_2D__
#define __Collision_Grid_2D__

#include <vector>

#include "SystemParams.h"
#include "A3DSquare.h"
#include "A3DVector.h"

#include <cmath>

typedef std::vector<int> GraphIndices;

class CollisionGrid3D
{
public:
	std::vector<A3DObject*>   _objects;
	std::vector<A3DSquare*>   _squares;
	std::vector<GraphIndices> _graphIndexArray;

	int   _side_num; // side length of the entire grid (unit: cell)
	float _max_cell_length; // side length of a cell

public:
	CollisionGrid3D();

	CollisionGrid3D(float cellSize);

	~CollisionGrid3D();	

	void InsertAPoint(float x, float y, float z, int info1, int info2); // make z positive

	void GetGraphIndices2B(float x, float y, float z, std::vector<int>& closestGraphIndices);

	void PrecomputeGraphIndices();

	void MovePoints();

private:
	void GetCellPosition(int& xPos, int& yPos, int& zPos, float x, float y, float z);

	int SquareIndex(int xPos, int yPos, int zPos);

};

#endif
