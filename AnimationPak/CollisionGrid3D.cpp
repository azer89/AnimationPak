
#include "CollisionGrid3D.h"
#include "UtilityFunctions.h"

CollisionGrid3D::CollisionGrid3D()
{
	// need to call init
}

void CollisionGrid3D::Init()
{
	_max_cell_length = SystemParams::_bin_square_size;

	// create squares, the grid is a square too
	_side_num = SystemParams::_upscaleFactor / _max_cell_length;

	//
	for (unsigned int z = 0; z < _side_num; z++) // fill the first layer...
	{
		for (unsigned int a = 0; a < _side_num; a++) // x -  fill the first column... then next column... repeat
		{
			for (unsigned int b = 0; b < _side_num; b++) // y - go down
			{
				float x_pos = a * _max_cell_length;
				float y_pos = b * _max_cell_length;
				float z_pos = z * _max_cell_length; // z is negative !!!
				_squares.push_back(new A3DSquare(x_pos, y_pos, z_pos, _max_cell_length));
			}
		}
	}

	// reserve
	_graphIndexArray.reserve(_squares.size());
	_triangleIndexArray.reserve(_squares.size());
}

/*CollisionGrid3D::CollisionGrid3D(float cellSize)
{
	_max_cell_length = cellSize;

	// create squares, the grid is a square too
	_side_num = SystemParams::_upscaleFactor / _max_cell_length;

	//
	for (unsigned int z = 0; z < _side_num; z++) // fill the first layer...
	{
		for (unsigned int a = 0; a < _side_num; a++) // x -  fill the first column... then next column... repeat
		{
			for (unsigned int b = 0; b < _side_num; b++) // y - go down
			{
				float x_pos = a * _max_cell_length;
				float y_pos = b * _max_cell_length;
				float z_pos = z * _max_cell_length; // z is negative !!!
				_squares.push_back(new A3DSquare(x_pos, y_pos, z_pos, _max_cell_length));
			}
		}
	}
}*/

CollisionGrid3D::~CollisionGrid3D()
{
	for (unsigned int a = 0; a < _squares.size(); a++)
	{
		_squares[a]->Clear();
		delete _squares[a];
	}
	_squares.clear();
}

int CollisionGrid3D::SquareIndex(int xPos, int yPos, int zPos)
{
	return (zPos * _side_num * _side_num) + (xPos * _side_num) + yPos;
}

void CollisionGrid3D::GetCellPosition(int& xPos, int& yPos, int& zPos, float x, float y, float z)
{
	xPos = x / _max_cell_length;
	yPos = y / _max_cell_length;
	zPos = z / _max_cell_length;

	if (xPos < 0) { xPos = 0; }
	else if (xPos >= _side_num) { xPos = _side_num - 1; }

	if (yPos < 0) { yPos = 0; }
	else if (yPos >= _side_num) { yPos = _side_num - 1; }

	if (zPos < 0) { zPos = 0; }
	else if (zPos >= _side_num) { zPos = _side_num - 1; }
}

void CollisionGrid3D::GetClosestPoints(float x, float y, float z, std::vector<A3DVector>& closestPoints)
{
	z = abs(z);

	if (x < 0 || x > SystemParams::_upscaleFactor ||
		y < 0 || y > SystemParams::_upscaleFactor ||
		z < 0 || z > SystemParams::_upscaleFactor)
	{
		return;
	}

	int xPos;
	int yPos;
	int zPos;
	GetCellPosition(xPos, yPos, zPos, x, y, z);

	int offst = SystemParams::_collission_block_radius;

	int xBegin = xPos - offst;
	if (xBegin < 0) { xBegin = 0; }

	int xEnd = xPos + offst;
	if (xEnd >= _side_num) { xEnd = _side_num - 1; }

	int yBegin = yPos - offst;
	if (yBegin < 0) { yBegin = 0; }

	int yEnd = yPos + offst;
	if (yEnd >= _side_num) { yEnd = _side_num - 1; }

	int zBegin = zPos - offst;
	if (zBegin < 0) { zBegin = 0; }

	int zEnd = zPos + offst;
	if (zEnd >= _side_num) { zEnd = _side_num - 1; }

	for (unsigned int zIter = zBegin; zIter <= xEnd; zIter++)
	{
		for (unsigned int xIter = xBegin; xIter <= xEnd; xIter++)
		{
			for (unsigned int yIter = yBegin; yIter <= yEnd; yIter++)
			{
				int idx = SquareIndex(xIter, yIter, zIter);

				for (unsigned int a = 0; a < _squares[idx]->_objects.size(); a++)
				{
					closestPoints.push_back(A3DVector(_squares[idx]->_objects[a]->_x,
						_squares[idx]->_objects[a]->_y,
						_squares[idx]->_objects[a]->_z));
				}
			}
		}
	}
}

void CollisionGrid3D::GetClosestObjects(float x, float y, float z, std::vector<A3DObject>& closestObjects)
{
	//if (std::isnan(x) || std::isinf(x) || std::isnan(y) || std::isinf(y))
	//{
	//	return;
	//}

	z = abs(z);

	if (x < 0 || x > SystemParams::_upscaleFactor || 
		y < 0 || y > SystemParams::_upscaleFactor || 
		z < 0 || z > SystemParams::_upscaleFactor)
	{
		return;
	}

	int xPos;
	int yPos;
	int zPos;
	GetCellPosition(xPos, yPos, zPos, x, y, z);

	int offst = SystemParams::_collission_block_radius;

	int xBegin = xPos - offst;
	if (xBegin < 0) { xBegin = 0; }

	int xEnd = xPos + offst;
	if (xEnd >= _side_num) { xEnd = _side_num - 1; }

	int yBegin = yPos - offst;
	if (yBegin < 0) { yBegin = 0; }

	int yEnd = yPos + offst;
	if (yEnd >= _side_num) { yEnd = _side_num - 1; }

	int zBegin = zPos - offst;
	if (zBegin < 0) { zBegin = 0; }

	int zEnd = zPos + offst;
	if (zEnd >= _side_num) { zEnd = _side_num - 1; }

	for (unsigned int zIter = zBegin; zIter <= xEnd; zIter++)
	{
		for (unsigned int xIter = xBegin; xIter <= xEnd; xIter++)
		{
			for (unsigned int yIter = yBegin; yIter <= yEnd; yIter++)
			{
				int idx = SquareIndex(xIter, yIter, zIter);

				for (unsigned int a = 0; a < _squares[idx]->_objects.size(); a++)
				{
					A3DObject obj3D(_squares[idx]->_objects[a]->_x, 
						            _squares[idx]->_objects[a]->_y, 
						            _squares[idx]->_objects[a]->_z, 
									_squares[idx]->_objects[a]->_info1, 
									_squares[idx]->_objects[a]->_info2);

					closestObjects.push_back(obj3D);
				}
			}
		}
	}
}


void CollisionGrid3D::GetTriangleIndices(float x, float y, float z, TriangleIndices& closestTriIndices)
{
	z = abs(z); // POSITIVE !!!

	int xPos;
	int yPos;
	int zPos;
	GetCellPosition(xPos, yPos, zPos, x, y, z);

	int idx = SquareIndex(xPos, yPos, zPos);

	closestTriIndices = _triangleIndexArray[idx];
}

void CollisionGrid3D::GetGraphIndices2B(float x, float y, float z, std::vector<int>& closestGraphIndices)
{
	z = abs(z); // POSITIVE !!!

	int xPos;
	int yPos;
	int zPos;
	GetCellPosition(xPos, yPos, zPos, x, y, z);

	int idx = SquareIndex(xPos, yPos, zPos);

	closestGraphIndices = _graphIndexArray[idx];
}

void CollisionGrid3D::SetPoint(int idx, A3DVector pos)
{
	pos._z = abs(pos._z);

	_objects[idx]->_x = pos._x;
	_objects[idx]->_y = pos._y;
	_objects[idx]->_z = pos._z;
}

void CollisionGrid3D::InsertAPoint(float x, float y, float z, int info1, int info2)
{
	z = abs(z); // POSITIVE !!!

	A3DObject* obj = new A3DObject(x, y, z, info1, info2);
	_objects.push_back(obj);
	int xPos;
	int yPos;
	int zPos;
	GetCellPosition(xPos, yPos, zPos, x, y, z);

	int sq_idx = SquareIndex(xPos, yPos, zPos);

	_squares[sq_idx]->_objects.push_back(obj);
}

void CollisionGrid3D::PrecomputeClosestGraphsAndTriangles()
{
	_graphIndexArray.clear(); // std::vector<std::vector<int>>

	int side_num_sq = _side_num * _side_num;

	for (unsigned int iter = 0; iter < _squares.size(); iter++)
	{
		GraphIndices gIndices;    // typedef std::vector<int> GraphIndices
		TriangleIndices tIndices; // typedef std::vector<std::vector<int>> TriangleIndices

		int zPos = iter / side_num_sq; // int division

		int left_over = iter - (zPos * side_num_sq);

		int xPos = left_over / _side_num;
		int yPos = left_over - (xPos * _side_num);

		int offst = SystemParams::_collission_block_radius;

		int xBegin = xPos - offst;
		if (xBegin < 0) { xBegin = 0; }

		int xEnd = xPos + offst;
		if (xEnd >= _side_num) { xEnd = _side_num - 1; }

		int yBegin = yPos - offst;
		if (yBegin < 0) { yBegin = 0; }

		int yEnd = yPos + offst;
		if (yEnd >= _side_num) { yEnd = _side_num - 1; }

		int zBegin = zPos - offst;
		if (zBegin < 0) { zBegin = 0; }

		int zEnd = zPos + offst;
		if (zEnd >= _side_num) { zEnd = _side_num - 1; }

		for (unsigned int zIter = zBegin; zIter <= xEnd; zIter++)
		{
			for (unsigned int xIter = xBegin; xIter <= xEnd; xIter++)
			{
				for (unsigned int yIter = yBegin; yIter <= yEnd; yIter++)
				{
					int idx = SquareIndex(xIter, yIter, zIter);
					for (unsigned int a = 0; a < _squares[idx]->_objects.size(); a++)
					{
						// graph						
						int info1 = _squares[idx]->_objects[a]->_info1;						
						int idxidx = UtilityFunctions::GetIndexFromIntList(gIndices, info1);
						if (idxidx == -1) // graph not found
						{
							// graph
							gIndices.push_back(info1);
							idxidx = gIndices.size() - 1;

							// triangle
							std::vector<int> tArray;
							tIndices.push_back(tArray);
							//tArray.push_back(info2); // don't assign now
							
						}

						// triangle
						int info2 = _squares[idx]->_objects[a]->_info2;
						int idxidx2 = UtilityFunctions::GetIndexFromIntList(tIndices[idxidx], info2);
						if (idxidx2 == -1)
						{
							tIndices[idxidx].push_back(info2);
						}


					}
				}
			}
		}
		_graphIndexArray.push_back(gIndices);
		_triangleIndexArray.push_back(tIndices);
	}
	std::cout << "done\n";
}

void CollisionGrid3D::MovePoints()
{
	std::vector<A3DObject*> invalidObjects;
	for (unsigned int a = 0; a < _squares.size(); a++)
	{
		for (int b = _squares[a]->_objects.size() - 1; b >= 0; b--) // should be signed
		{
			if (!_squares[a]->Contains(_squares[a]->_objects[b]))
			{
				invalidObjects.push_back(_squares[a]->_objects[b]);
				_squares[a]->_objects.erase(_squares[a]->_objects.begin() + b);
			}
		}
	}

	for (unsigned int a = 0; a < invalidObjects.size(); a++)
	{
		int xPos;
		int yPos;
		int zPos;
		GetCellPosition(xPos, yPos, zPos, invalidObjects[a]->_x, invalidObjects[a]->_y, invalidObjects[a]->_z);

		// avoiding runtime error
		if (xPos < 0) { xPos = 0; }
		if (xPos == _side_num) { xPos = _side_num - 1; }

		if (yPos < 0) { yPos = 0; }
		if (yPos == _side_num) { yPos = _side_num - 1; }

		if (zPos < 0) { zPos = 0; }
		if (zPos == _side_num) { zPos = _side_num - 1; }

		int idx = SquareIndex(xPos, yPos, zPos);
		_squares[idx]->_objects.push_back(invalidObjects[a]);
	}
	invalidObjects.clear();
}
