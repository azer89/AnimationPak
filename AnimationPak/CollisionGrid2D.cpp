
#include "CollisionGrid2D.h"
#include "UtilityFunctions.h"


CollisionGrid2D::CollisionGrid2D()
{
	_maxLength = SystemParams::_bin_square_size;

	// create squares, the grid is a square too
	_numColumn = SystemParams::_upscaleFactor / _maxLength;

	// <= is important !!!
	for (unsigned int a = 0; a < _numColumn; a++) // x -  fill the first column... then next column... repeat
	{
		for (unsigned int b = 0; b < _numColumn; b++) // y - go down
		{
			float x = a * _maxLength;
			float y = b * _maxLength;
			_squares.push_back(new A2DSquare(x, y, _maxLength));
		}
	}
}

CollisionGrid2D::CollisionGrid2D(float cellSize)
{
	_maxLength = cellSize;

	// create squares, the grid is a square too
	_numColumn = SystemParams::_upscaleFactor / _maxLength;

	// <= is important !!!
	for (unsigned int a = 0; a < _numColumn; a++) // x -  fill the first column... then next column... repeat
	{
		for (unsigned int b = 0; b < _numColumn; b++) // y - go down
		{
			float x = a * _maxLength;
			float y = b * _maxLength;
			_squares.push_back(new A2DSquare(x, y, _maxLength));
		}
	}
}

CollisionGrid2D::~CollisionGrid2D()
{
	for (unsigned int a = 0; a < _squares.size(); a++)
	{
		_squares[a]->Clear();
		delete _squares[a];
	}
	_squares.clear();
}

void CollisionGrid2D::GetCellPosition(int& xPos, int& yPos, float x, float y)
{
	xPos = x / _maxLength;
	yPos = y / _maxLength;
	if (xPos < 0) { xPos = 0; }
	else if (xPos >= _numColumn) { xPos = _numColumn - 1; }

	if (yPos < 0) { yPos = 0; }
	else if (yPos >= _numColumn) { yPos = _numColumn - 1; }
}

/*
void CollissionGrid::InsertAPoint(float x, float y, int info1, int info2, float nx, float ny)
{
AnObject* obj = new AnObject(x, y, info1, info2, nx, ny);
_objects.push_back(obj);

//int xPos = x / _maxLength;
//int yPos = y / _maxLength;
int xPos;
int yPos;
GetCellPosition(xPos, yPos, x, y);

_squares[(xPos * _numColumn) + yPos]->_objects.push_back(obj);
}*/

void CollisionGrid2D::InsertAPoint(float x, float y, int info1, int info2)
{
	A2DObject* obj = new A2DObject(x, y, info1, info2);
	_objects.push_back(obj);

	//int xPos = x / _maxLength;
	//int yPos = y / _maxLength;
	int xPos;
	int yPos;
	GetCellPosition(xPos, yPos, x, y);

	_squares[(xPos * _numColumn) + yPos]->_objects.push_back(obj);
}


void CollisionGrid2D::PrecomputeGraphIndices()
{
	//std::vector<ASquare*> _squares;
	// int _numColumn;
	// std::vector<GraphIndices> _graphIndexArray;

	_graphIndexArray.clear(); // std::vector<std::vector<int>>

	for (unsigned int iter = 0; iter < _squares.size(); iter++)
	{
		GraphIndices gIndices; // typedef std::vector<int> GraphIndices

		int xPos = iter / _numColumn;
		int yPos = iter - (xPos * _numColumn);

		int offst = SystemParams::_grid_radius_1;

		int xBegin = xPos - offst;
		if (xBegin < 0) { xBegin = 0; }

		int xEnd = xPos + offst;
		if (xEnd >= _numColumn) { xEnd = _numColumn - 1; }

		int yBegin = yPos - offst;
		if (yBegin < 0) { yBegin = 0; }

		int yEnd = yPos + offst;
		if (yEnd >= _numColumn) { yEnd = _numColumn - 1; }

		for (unsigned int xIter = xBegin; xIter <= xEnd; xIter++)
		{
			for (unsigned int yIter = yBegin; yIter <= yEnd; yIter++)
			{
				int idx = (xIter * _numColumn) + yIter;
				for (unsigned int a = 0; a < _squares[idx]->_objects.size(); a++)
				{
					int info1 = _squares[idx]->_objects[a]->_info1;
					if (UtilityFunctions::GetIndexFromIntList(gIndices, info1) == -1)
					{
						gIndices.push_back(info1);
					}
				}
			}
		}

		_graphIndexArray.push_back(gIndices);
	}
}

void CollisionGrid2D::GetGraphIndices1B(float x, float y, std::vector<int>& closestGraphIndices)
{
	if (x < 0 || x > SystemParams::_upscaleFactor || y < 0 || y > SystemParams::_upscaleFactor) { return; }

	int xPos;
	int yPos;
	GetCellPosition(xPos, yPos, x, y);

	int idx = (xPos * _numColumn) + yPos;

	if (_graphIndexArray[idx].size() > 0)
		closestGraphIndices = _graphIndexArray[idx];
}

void CollisionGrid2D::GetGraphIndices2B(float x, float y, std::vector<int>& closestGraphIndices)
{
	// warning !!! no check
	//if (std::isnan(x) || std::isinf(x) || std::isnan(y) || std::isinf(y)) { return; }
	//if (x < 0 || x > SystemParams::_upscaleFactor || y < 0 || y > SystemParams::_upscaleFactor) { return; }

	int xPos;
	int yPos;
	GetCellPosition(xPos, yPos, x, y);

	int idx = (xPos * _numColumn) + yPos;

	//if (_graphIndexArray[idx].size() > 0)
	//{
	closestGraphIndices = _graphIndexArray[idx];
	//}
	//else
	//{
	//	return;
	//}

	/*for (unsigned int a = 0; a < closestGraphIndices.size(); a++)
	{
	if (closestGraphIndices[a] == parentGraphIndex)
	{
	closestGraphIndices.erase(closestGraphIndices.begin() + a);
	break;
	}
	}*/
}

void CollisionGrid2D::GetGraphIndices1(float x, float y, std::vector<int>& closestGraphIndices)
{
	if (std::isnan(x) || std::isinf(x) || std::isnan(y) || std::isinf(y)) { return; }

	if (x < 0 || x > SystemParams::_upscaleFactor || y < 0 || y > SystemParams::_upscaleFactor) { return; }


	int xPos;
	int yPos;
	GetCellPosition(xPos, yPos, x, y);

	int offst = SystemParams::_grid_radius_1;

	int xBegin = xPos - offst;
	if (xBegin < 0) { xBegin = 0; }

	int xEnd = xPos + offst;
	if (xEnd >= _numColumn) { xEnd = _numColumn - 1; }

	int yBegin = yPos - offst;
	if (yBegin < 0) { yBegin = 0; }

	int yEnd = yPos + offst;
	if (yEnd >= _numColumn) { yEnd = _numColumn - 1; }

	for (unsigned int xIter = xBegin; xIter <= xEnd; xIter++)
	{
		for (unsigned int yIter = yBegin; yIter <= yEnd; yIter++)
		{
			int idx = (xIter * _numColumn) + yIter;

			for (unsigned int a = 0; a < _squares[idx]->_objects.size(); a++)
			{
				int info1 = _squares[idx]->_objects[a]->_info1;
				if (UtilityFunctions::GetIndexFromIntList(closestGraphIndices, info1) == -1)
				{
					closestGraphIndices.push_back(info1);
				}
			}
		}
	}
}

void CollisionGrid2D::GetGraphIndices2(float x, float y, int parentGraphIndex, std::vector<int>& closestGraphIndices)
{
	//std::vector<AnObject*> nearObjects;

	if (std::isnan(x) || std::isinf(x) || std::isnan(y) || std::isinf(y)) { return; }
	if (x < 0 || x > SystemParams::_upscaleFactor || y < 0 || y > SystemParams::_upscaleFactor) { return; }

	//int xPos = x / _maxLength;
	//int yPos = y / _maxLength;
	int xPos;
	int yPos;
	GetCellPosition(xPos, yPos, x, y);


	int offst = SystemParams::_grid_radius_1;

	int xBegin = xPos - offst;
	if (xBegin < 0) { xBegin = 0; }

	int xEnd = xPos + offst;
	if (xEnd >= _numColumn) { xEnd = _numColumn - 1; }

	int yBegin = yPos - offst;
	if (yBegin < 0) { yBegin = 0; }

	int yEnd = yPos + offst;
	if (yEnd >= _numColumn) { yEnd = _numColumn - 1; }

	for (unsigned int xIter = xBegin; xIter <= xEnd; xIter++)
	{
		for (unsigned int yIter = yBegin; yIter <= yEnd; yIter++)
		{
			int idx = (xIter * _numColumn) + yIter;

			for (unsigned int a = 0; a < _squares[idx]->_objects.size(); a++)
			{
				int info1 = _squares[idx]->_objects[a]->_info1;

				if (info1 != parentGraphIndex)
				{
					if (UtilityFunctions::GetIndexFromIntList(closestGraphIndices, info1) == -1)
					{
						closestGraphIndices.push_back(info1);
					}
				}
			}
		}
	}
}
void  CollisionGrid2D::GetClosestPoints(float x, float y, std::vector<A2DVector>& closestPts)
{
	if (std::isnan(x) || std::isinf(x) || std::isnan(y) || std::isinf(y))
	{
		return;
	}

	if (x < 0 || x > SystemParams::_upscaleFactor || y < 0 || y > SystemParams::_upscaleFactor)
	{
		return;
	}

	//int xPos = x / _maxLength;
	//int yPos = y / _maxLength;
	int xPos;
	int yPos;
	GetCellPosition(xPos, yPos, x, y);

	int offst = SystemParams::_grid_radius_1;

	int xBegin = xPos - offst;
	if (xBegin < 0) { xBegin = 0; }

	int xEnd = xPos + offst;
	if (xEnd >= _numColumn) { xEnd = _numColumn - 1; }

	int yBegin = yPos - offst;
	if (yBegin < 0) { yBegin = 0; }

	int yEnd = yPos + offst;
	if (yEnd >= _numColumn) { yEnd = _numColumn - 1; }

	for (unsigned int xIter = xBegin; xIter <= xEnd; xIter++)
	{
		for (unsigned int yIter = yBegin; yIter <= yEnd; yIter++)
		{
			int idx = (xIter * _numColumn) + yIter;

			for (unsigned int a = 0; a < _squares[idx]->_objects.size(); a++)
			{

				A2DVector pt;
				pt.x = _squares[idx]->_objects[a]->_x;
				pt.y = _squares[idx]->_objects[a]->_y;
				closestPts.push_back(pt);
			}

			//nearObjects.insert(nearObjects.end(), _squares[idx]->_objects.begin(), _squares[idx]->_objects.end());

			//std::vector<AnObject*> objs = _squares[(xIter * _numColumn) + yIter]->_objects;
			//nearObjects.insert(nearObjects.end(), objs.begin(), objs.end());
		}
	}
}

void CollisionGrid2D::GetData(float x, float y, int parentGraphIndex, std::vector<A2DVector>& closestPts, std::vector<int>& closestGraphIndices)
{
	//std::vector<AnObject*> nearObjects;

	if (std::isnan(x) || std::isinf(x) || std::isnan(y) || std::isinf(y))
	{
		return;
	}

	if (x < 0 || x > SystemParams::_upscaleFactor || y < 0 || y > SystemParams::_upscaleFactor)
	{
		return;
	}

	//int xPos = x / _maxLength;
	//int yPos = y / _maxLength;
	int xPos;
	int yPos;
	GetCellPosition(xPos, yPos, x, y);

	int offst = SystemParams::_grid_radius_1;

	int xBegin = xPos - offst;
	if (xBegin < 0) { xBegin = 0; }

	int xEnd = xPos + offst;
	if (xEnd >= _numColumn) { xEnd = _numColumn - 1; }

	int yBegin = yPos - offst;
	if (yBegin < 0) { yBegin = 0; }

	int yEnd = yPos + offst;
	if (yEnd >= _numColumn) { yEnd = _numColumn - 1; }

	for (unsigned int xIter = xBegin; xIter <= xEnd; xIter++)
	{
		for (unsigned int yIter = yBegin; yIter <= yEnd; yIter++)
		{
			int idx = (xIter * _numColumn) + yIter;

			for (unsigned int a = 0; a < _squares[idx]->_objects.size(); a++)
			{
				int info1 = _squares[idx]->_objects[a]->_info1;

				if (info1 != parentGraphIndex)
				{
					if (UtilityFunctions::GetIndexFromIntList(closestGraphIndices, info1) == -1)
					{
						closestGraphIndices.push_back(info1);
					}

					A2DVector pt;
					pt.x = _squares[idx]->_objects[a]->_x;
					pt.y = _squares[idx]->_objects[a]->_y;
					closestPts.push_back(pt);
				}
			}

			//nearObjects.insert(nearObjects.end(), _squares[idx]->_objects.begin(), _squares[idx]->_objects.end());

			//std::vector<AnObject*> objs = _squares[(xIter * _numColumn) + yIter]->_objects;
			//nearObjects.insert(nearObjects.end(), objs.begin(), objs.end());
		}
	}
}

std::vector<A2DObject*> CollisionGrid2D::GetObjects(float x, float y)
{
	std::vector<A2DObject*> nearObjects;

	if (std::isnan(x) || std::isinf(x) || std::isnan(y) || std::isinf(y))
	{
		return nearObjects;
	}

	if (x < 0 || x > SystemParams::_upscaleFactor || y < 0 || y > SystemParams::_upscaleFactor)
	{
		return nearObjects;
	}

	//int xPos = x / _maxLength;
	//int yPos = y / _maxLength;
	int xPos;
	int yPos;
	GetCellPosition(xPos, yPos, x, y);

	int offst = SystemParams::_grid_radius_1;

	int xBegin = xPos - offst;
	if (xBegin < 0) { xBegin = 0; }

	int xEnd = xPos + offst;
	if (xEnd >= _numColumn) { xEnd = _numColumn - 1; }

	int yBegin = yPos - offst;
	if (yBegin < 0) { yBegin = 0; }

	int yEnd = yPos + offst;
	if (yEnd >= _numColumn) { yEnd = _numColumn - 1; }

	for (unsigned int xIter = xBegin; xIter <= xEnd; xIter++)
	{
		for (unsigned int yIter = yBegin; yIter <= yEnd; yIter++)
		{
			//if (xIter < 0 || xIter >= _numColumn || yIter < 0 || yIter >= _numColumn) { continue; }

			int idx = (xIter * _numColumn) + yIter;

			nearObjects.insert(nearObjects.end(), _squares[idx]->_objects.begin(), _squares[idx]->_objects.end());

			//std::vector<AnObject*> objs = _squares[(xIter * _numColumn) + yIter]->_objects;
			//nearObjects.insert(nearObjects.end(), objs.begin(), objs.end());
		}
	}

	return nearObjects;
}

void CollisionGrid2D::MovePoints()
{
	std::vector<A2DObject*> invalidObjects;
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
		//int xPos = invalidObjects[a]->_x / _maxLength;
		//int yPos = invalidObjects[a]->_y / _maxLength;
		int xPos;
		int yPos;
		GetCellPosition(xPos, yPos, invalidObjects[a]->_x, invalidObjects[a]->_y);
		
		// avoiding runtime error
		if (xPos < 0) { xPos = 0; }
		if (xPos == _numColumn) { xPos = _numColumn - 1; }
		if (yPos < 0) { yPos = 0; }
		if (yPos == _numColumn) { yPos = _numColumn - 1; }

		_squares[(xPos * _numColumn) + yPos]->_objects.push_back(invalidObjects[a]);
	}
	invalidObjects.clear();
}

/*void CollisionGrid::Draw()
{
	for (unsigned int a = 0; a < _squares.size(); a++)
	{
		_squares[a]->Draw();
	}
}*/

/*bool CollisionGrid::NearBoundary(float x, float y)
{
	//int xPos = x / _maxLength;
	//int yPos = y / _maxLength;
	int xPos;
	int yPos;
	GetCellPosition(xPos, yPos, x, y);

	return (_squares[(xPos * _numColumn) + yPos]->_containerFlag == 0);
}*/

/*void CollisionGrid::AnalyzeContainer(const std::vector<std::vector<A2DVector>>& boundaries, 
                                       const std::vector<std::vector<A2DVector>>& holes, 
									   const std::vector<std::vector<A2DVector>>& offsetFocalBoundaries)
{
	//std::cout << "_maxLength " << _maxLength << "\n";

	for (unsigned int a = 0; a < _squares.size(); a++)
	{
		// ===== MULTIPLE CONTAINERS =====
		// 1 UNCOMMENT THIS FOR HOLES		
		float d1 = UtilityFunctions::DistanceToClosedCurves(boundaries, AVector(_squares[a]->_xCenter, _squares[a]->_yCenter));
		float d2 = UtilityFunctions::DistanceToClosedCurves(offsetFocalBoundaries, AVector(_squares[a]->_xCenter, _squares[a]->_yCenter));
		float d3 = UtilityFunctions::DistanceToClosedCurves(holes, AVector(_squares[a]->_xCenter, _squares[a]->_yCenter));
		if (d1 > _maxLength && d2 > _maxLength && d3 > _maxLength)
		{
			_squares[a]->_containerFlag = 1;
		}

		// 2 UNCOMMENT THIS FOR MULTIPLE BOUNDARIES
		//float d1 = UtilityFunctions::DistanceToClosedCurves(boundaries, AVector(_squares[a]->_xCenter, _squares[a]->_yCenter));
		//if (d1 > _maxLength) { _squares[a]->_containerFlag = 1; }
	}
}*/