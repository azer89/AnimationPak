
#include "CollisionGrid3D.h"
#include "UtilityFunctions.h"

#include <math.h>

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

	// reserve _idx_array
	_graph_idx_array.reserve(_squares.size());
	_triangle_idx_array.reserve(_squares.size());
	_approx_graph_idx_array.reserve(_squares.size());
	_approx_triangle_idx_array.reserve(_squares.size());


	// pair data
	//_pair_data_array.reserve(_squares.size());
	//_approx_pair_data_array.reserve(_squares.size());
	//_pd.reserve(_squares.size());
	//_approx_pd.reserve(_squares.size());
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

	int offst = SystemParams::_grid_radius_1;

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

	int offst = SystemParams::_grid_radius_1;

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

	// _idx_array
	closestTriIndices = _triangle_idx_array[idx];
}

int CollisionGrid3D::GetSquareIndexFromFloat(float x, float y, float z)
{
	z = abs(z); // POSITIVE !!!

	int xPos;
	int yPos;
	int zPos;
	GetCellPosition(xPos, yPos, zPos, x, y, z);

	return SquareIndex(xPos, yPos, zPos);
}

void CollisionGrid3D::GetData(float x, float y, float z, PairData& pair_data_array, PairData& approx_pair_data_array)
{
	z = abs(z); // POSITIVE !!!

	int xPos;
	int yPos;
	int zPos;
	GetCellPosition(xPos, yPos, zPos, x, y, z);

	int idx = SquareIndex(xPos, yPos, zPos);

	std::cout << "error!";
	// _idx_array
	//pair_data_array = _pair_data_array[idx];
	//approx_pair_data_array = _approx_pair_data_array[idx];
}

void CollisionGrid3D::GetGraphIndices2B(float x, float y, float z, std::vector<int>& closestGraphIndices)
{
	z = abs(z); // POSITIVE !!!

	int xPos;
	int yPos;
	int zPos;
	GetCellPosition(xPos, yPos, zPos, x, y, z);

	int idx = SquareIndex(xPos, yPos, zPos);

	// _idx_array
	closestGraphIndices = _graph_idx_array[idx];
}

void CollisionGrid3D::SetPoint(int idx, A3DVector pos)
{
	pos._z = abs(pos._z); // positive

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

void CollisionGrid3D::PrecomputeData()
{
	//_pair_data_array.clear();
	//_approx_pair_data_array.clear();
//
	int side_num_sq = _side_num * _side_num;
	int offst = SystemParams::_grid_radius_2;

	A3DSquare* cur_sq;
	A3DSquare* neighbor_sq;

	for (unsigned int iter = 0; iter < _squares.size(); iter++)
	{
		cur_sq = _squares[iter];
		
		if (cur_sq->_objects.size() == 0) {continue;}
		
		cur_sq->_c_pt_fill_size = 0;        // reset
		cur_sq->_c_pt_approx_fill_size = 0; // reset

		int zPos = iter / side_num_sq; // int division

		int left_over = iter - (zPos * side_num_sq);

		int xPos = left_over / _side_num;          // current position
		int yPos = left_over - (xPos * _side_num); // current position
			
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

		for (int zIter = zBegin; zIter <= xEnd; zIter++)
		{
			for (int xIter = xBegin; xIter <= xEnd; xIter++)
			{
				for (int yIter = yBegin; yIter <= yEnd; yIter++)
				{
					int s_idx = SquareIndex(xIter, yIter, zIter);
					neighbor_sq = _squares[s_idx];
					if (abs(xIter - xPos) <= SystemParams::_grid_radius_1 &&
						abs(yIter - yPos) <= SystemParams::_grid_radius_1 &&
						abs(zIter - zPos) <= SystemParams::_grid_radius_1)
					{
						for (unsigned int a = 0; a < neighbor_sq->_objects.size(); a++)
						{ 
							// (1) which element (2) which triangle
							cur_sq->_c_pt[cur_sq->_c_pt_fill_size] = std::pair<int, int>(neighbor_sq->_objects[a]->_info1, neighbor_sq->_objects[a]->_info2);
							cur_sq->_c_pt_fill_size++;

							/*if (cur_sq->_c_pt_fill_size > SystemParams::_max_exact_array_len)
							{
								std::cout << "e";
							}*/
						}
					}
					//else if (abs(xIter - xPos) <= SystemParams::_grid_radius_2 &&
					//	     abs(yIter - yPos) <= SystemParams::_grid_radius_2 &&
					//	     abs(zIter - zPos) <= SystemParams::_grid_radius_2)
					else // it is ok, no need to check
					{				
						for (unsigned int a = 0; a < neighbor_sq->_objects.size(); a++)
						{
							// (1) which element (2) which square
							cur_sq->_c_pt_approx[cur_sq->_c_pt_approx_fill_size] = std::pair<int, int>(neighbor_sq->_objects[a]->_info1,
								                                                                 s_idx);
							cur_sq->_c_pt_approx_fill_size++;

							/*if (cur_sq->_c_pt_approx_fill_size > SystemParams::_max_approx_array_len)
							{
								std::cout << "a";
							}*/
						}
					}					
				}
			}
		}
		//std::cout << cur_sq->_pd_actual_size << " -- " << cur_sq->_approx_pd_actual_size << "\n";
		//_pair_data_array.push_back(_pd);
		//_approx_pair_data_array.push_back(_approx_pd);
		//_graph_idx_array.push_back(gIndices);
		//_triangle_idx_array.push_back(tIndices);
	}
}

void CollisionGrid3D::PrecomputeClosestGraphsAndTriangles()
{
	_graph_idx_array.clear(); // std::vector<std::vector<int>>
	_triangle_idx_array.clear();

	int side_num_sq = _side_num * _side_num;

	for (unsigned int iter = 0; iter < _squares.size(); iter++)
	{
		GraphIndices gIndices;    // typedef std::vector<int> GraphIndices
		TriangleIndices tIndices; // typedef std::vector<std::vector<int>> TriangleIndices

		int zPos = iter / side_num_sq; // int division

		int left_over = iter - (zPos * side_num_sq);

		int xPos = left_over / _side_num;          // current position
		int yPos = left_over - (xPos * _side_num); // current position

		int offst = SystemParams::_grid_radius_1;

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

						// can be improved, put in an temp array
						// triangle
						int info2 = _squares[idx]->_objects[a]->_info2;
						int idxidx2 = UtilityFunctions::GetIndexFromIntList(tIndices[idxidx], info2); // actually they don't have duplicates
						if (idxidx2 == -1)
						{
							tIndices[idxidx].push_back(info2);
						}


					}
				}
			}
		}
		_graph_idx_array.push_back(gIndices);
		_triangle_idx_array.push_back(tIndices);
	}
	//std::cout << "done\n";
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

void CollisionGrid3D::InitOgre3D(Ogre::SceneManager* sceneMgr)
{
	// showing the amount of c_pt_approx in each square
	//DynamicLines*    _c_pt_approx_lines;
	//Ogre::SceneNode* _c_pt_approx_node;
	Ogre::MaterialPtr c_pt_approx_material = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("CGCPT-Approx");
	c_pt_approx_material->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(0, 0, 1, 1));
	_c_pt_approx_lines = new DynamicLines(c_pt_approx_material, Ogre::RenderOperation::OT_LINE_LIST);
	/*for (unsigned int a = 0; a < _squares.size(); a++)
	{
		if (_squares[a]->_objects.size() > 0)
		{
			A3DVector pos(_squares[a]->_xCenter, _squares[a]->_yCenter, -_squares[a]->_zCenter);

			_c_pt_approx_lines->addPoint(pos._x - 0, pos._y, pos._z);
			_c_pt_approx_lines->addPoint(pos._x + 0, pos._y, pos._z);
			_c_pt_approx_lines->addPoint(pos._x, pos._y - 0, pos._z);
			_c_pt_approx_lines->addPoint(pos._x, pos._y + 0, pos._z);
		}
	}*/
	_c_pt_approx_lines->update();
	_c_pt_approx_node = sceneMgr->getRootSceneNode()->createChildSceneNode("c_pt_approx_cg_node");
	_c_pt_approx_node->attachObject(_c_pt_approx_lines);

	// showing the amount of c_pt in each square
	//DynamicLines*    _c_pt_lines;
	//Ogre::SceneNode* _c_pt_node;
	Ogre::MaterialPtr c_pt_material = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("CGCPT");
	c_pt_material->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(1, 0, 0, 1));
	_c_pt_lines = new DynamicLines(c_pt_material, Ogre::RenderOperation::OT_LINE_LIST);
	/*for (unsigned int a = 0; a < _squares.size(); a++)
	{
		if (_squares[a]->_objects.size() > 0)
		{
			A3DVector pos(_squares[a]->_xCenter, _squares[a]->_yCenter, -_squares[a]->_zCenter);

			_c_pt_lines->addPoint(pos._x - 0, pos._y, pos._z);
			_c_pt_lines->addPoint(pos._x + 0, pos._y, pos._z);
			_c_pt_lines->addPoint(pos._x, pos._y - 0, pos._z);
			_c_pt_lines->addPoint(pos._x, pos._y + 0, pos._z);
		}
	}*/
	_c_pt_lines->update();
	_c_pt_node = sceneMgr->getRootSceneNode()->createChildSceneNode("c_pt_cg_node");
	_c_pt_node->attachObject(_c_pt_lines);

	

	// object viz
	Ogre::MaterialPtr plus_material = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("PlusColissionGridLines");
	plus_material->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(1, 0, 0, 1));
	_plus_lines = new DynamicLines(plus_material, Ogre::RenderOperation::OT_LINE_LIST);
	/*for (unsigned int a = 0; a < _squares.size(); a++)
	{
		if (_squares[a]->_objects.size() > 0)
		{
			for (unsigned int b = 0; b < _squares[a]->_objects.size(); b++)
			{
				A3DVector pos(_squares[a]->_objects[b]->_x, _squares[a]->_objects[b]->_y, -_squares[a]->_objects[b]->_z);

				_plus_lines->addPoint(pos._x - 0.5, pos._y, pos._z);
				_plus_lines->addPoint(pos._x + 0.5, pos._y, pos._z);
				_plus_lines->addPoint(pos._x, pos._y - 0.5, pos._z);
				_plus_lines->addPoint(pos._x, pos._y + 0.5, pos._z);
			}
		}
	}*/
	_plus_lines->update();
	_plus_node = sceneMgr->getRootSceneNode()->createChildSceneNode("plus_lines_node");
	_plus_node->attachObject(_plus_lines);

	// box viz
	Ogre::MaterialPtr filled_material = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("FilledColissionGridLines");
	filled_material->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(0, 1, 0, 1));
	_filled_lines = new DynamicLines(filled_material, Ogre::RenderOperation::OT_LINE_LIST);
	_filled_lines->update();
	_filled_node = sceneMgr->getRootSceneNode()->createChildSceneNode("filled_lines_node");
	_filled_node->attachObject(_filled_lines);


	//_plus_node->showBoundingBox(true);
	//_filled_node->showBoundingBox(true);
	//_plus_lines->setBoundingBox(Ogre::AxisAlignedBox(-1000000, -1000000, -1000000, 1000000, 1000000, 1000000));
	//_filled_lines->setBoundingBox(Ogre::AxisAlignedBox(-1000000, -1000000, -1000000, 1000000, 1000000, 1000000));
	_plus_lines->setBoundingBox(Ogre::AxisAlignedBox(0, 0, 0, 1, 1, 1));
	_filled_lines->setBoundingBox(Ogre::AxisAlignedBox(0, 0, 0, 1, 1, 1));
	// ::EXTENT_INFINITE
	//_plus_lines->setBoundingBox(Ogre::AxisAlignedBox::EXTENT_INFINITE);
	//_filled_lines->setBoundingBox(Ogre::AxisAlignedBox::EXTENT_INFINITE);
	
	
}

void CollisionGrid3D::UpdateOgre3D()
{
	_plus_lines->clear();
	_filled_lines->clear();

	_c_pt_approx_lines->clear();
	_c_pt_lines->clear();

	float plus_offset = 0.5;
	//int plus_iter = 0;

	// do something
	//int empty_iter = 0;
	//int filled_iter = 0;
	if (SystemParams::_show_collision_grid_object)
	{
		for (unsigned int a = 0; a < _squares.size(); a++)
		{
			if (_squares[a]->_objects.size() > 0)
			{
				for (unsigned int b = 0; b < _squares[a]->_objects.size(); b++)
				{
					A3DVector pos(_squares[a]->_objects[b]->_x, _squares[a]->_objects[b]->_y, -_squares[a]->_objects[b]->_z);

					_plus_lines->addPoint(pos._x - plus_offset, pos._y, pos._z);
					_plus_lines->addPoint(pos._x + plus_offset, pos._y, pos._z);
					_plus_lines->addPoint(pos._x, pos._y - plus_offset, pos._z);
					_plus_lines->addPoint(pos._x, pos._y + plus_offset, pos._z);

					/*_plus_lines->setPoint(plus_iter++, Ogre::Vector3(pos._x - plus_offset, pos._y, pos._z));
					_plus_lines->setPoint(plus_iter++, Ogre::Vector3(pos._x + plus_offset, pos._y, pos._z));
					_plus_lines->setPoint(plus_iter++, Ogre::Vector3(pos._x, pos._y - plus_offset, pos._z));
					_plus_lines->setPoint(plus_iter++, Ogre::Vector3(pos._x, pos._y + plus_offset, pos._z));*/
				}
			}
		}
	}


	if (SystemParams::_show_collision_grid)
	{
		//_plus_node->setVisible(true);
		//_filled_node->setVisible(true);
		for (unsigned int a = 0; a < _squares.size(); a++)
		{
			if (_squares[a]->_objects.size() > 0)
			{
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt1._x, _squares[a]->_draw_pt1._y, -_squares[a]->_draw_pt1._z));//
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt2._x, _squares[a]->_draw_pt2._y, -_squares[a]->_draw_pt2._z));
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt2._x, _squares[a]->_draw_pt2._y, -_squares[a]->_draw_pt2._z));//
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt4._x, _squares[a]->_draw_pt4._y, -_squares[a]->_draw_pt4._z));
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt3._x, _squares[a]->_draw_pt3._y, -_squares[a]->_draw_pt3._z));//
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt4._x, _squares[a]->_draw_pt4._y, -_squares[a]->_draw_pt4._z));
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt1._x, _squares[a]->_draw_pt1._y, -_squares[a]->_draw_pt1._z));//
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt3._x, _squares[a]->_draw_pt3._y, -_squares[a]->_draw_pt3._z));

				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt5._x, _squares[a]->_draw_pt5._y, -_squares[a]->_draw_pt5._z));//
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt6._x, _squares[a]->_draw_pt6._y, -_squares[a]->_draw_pt6._z));
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt6._x, _squares[a]->_draw_pt6._y, -_squares[a]->_draw_pt6._z));//
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt8._x, _squares[a]->_draw_pt8._y, -_squares[a]->_draw_pt8._z));
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt7._x, _squares[a]->_draw_pt7._y, -_squares[a]->_draw_pt7._z));//
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt8._x, _squares[a]->_draw_pt8._y, -_squares[a]->_draw_pt8._z));
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt5._x, _squares[a]->_draw_pt5._y, -_squares[a]->_draw_pt5._z));//
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt7._x, _squares[a]->_draw_pt7._y, -_squares[a]->_draw_pt7._z));

				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt1._x, _squares[a]->_draw_pt1._y, -_squares[a]->_draw_pt1._z));//
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt5._x, _squares[a]->_draw_pt5._y, -_squares[a]->_draw_pt5._z));
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt2._x, _squares[a]->_draw_pt2._y, -_squares[a]->_draw_pt2._z));//
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt6._x, _squares[a]->_draw_pt6._y, -_squares[a]->_draw_pt6._z));
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt4._x, _squares[a]->_draw_pt4._y, -_squares[a]->_draw_pt4._z));//
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt8._x, _squares[a]->_draw_pt8._y, -_squares[a]->_draw_pt8._z));
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt3._x, _squares[a]->_draw_pt3._y, -_squares[a]->_draw_pt3._z));//
				_filled_lines->addPoint(Ogre::Vector3(_squares[a]->_draw_pt7._x, _squares[a]->_draw_pt7._y, -_squares[a]->_draw_pt7._z));
			
			
				
			}

		}

	}

	if (SystemParams::_show_c_pt_cg)
	{
		for (unsigned int a = 0; a < _squares.size(); a++)
		{
			if (_squares[a]->_objects.size() > 0)
			{
				float offsetVal = _squares[a]->_length * 0.5 * (float)_squares[a]->_c_pt_fill_size / (float)SystemParams::_max_exact_array_len;

				A3DVector pos(_squares[a]->_xCenter, _squares[a]->_yCenter, -_squares[a]->_zCenter);

				_c_pt_lines->addPoint(Ogre::Vector3(pos._x - offsetVal, pos._y, pos._z));
				_c_pt_lines->addPoint(Ogre::Vector3(pos._x + offsetVal, pos._y, pos._z));
				_c_pt_lines->addPoint(Ogre::Vector3(pos._x, pos._y - offsetVal, pos._z));
				_c_pt_lines->addPoint(Ogre::Vector3(pos._x, pos._y + offsetVal, pos._z));
			}
		}
	}

	if (SystemParams::_show_c_pt_approx_cg)
	{
		for (unsigned int a = 0; a < _squares.size(); a++)
		{
			if (_squares[a]->_objects.size() > 0)
			{
				float offsetVal = _squares[a]->_length * 0.5 * (float)_squares[a]->_c_pt_approx_fill_size / (float)SystemParams::_max_exact_array_len;

				A3DVector pos(_squares[a]->_xCenter, _squares[a]->_yCenter, -_squares[a]->_zCenter);

				_c_pt_approx_lines->addPoint(Ogre::Vector3(pos._x - offsetVal, pos._y, pos._z));
				_c_pt_approx_lines->addPoint(Ogre::Vector3(pos._x + offsetVal, pos._y, pos._z));
				_c_pt_approx_lines->addPoint(Ogre::Vector3(pos._x, pos._y - offsetVal, pos._z));
				_c_pt_approx_lines->addPoint(Ogre::Vector3(pos._x, pos._y + offsetVal, pos._z));
			}
		}
	}

	_plus_lines->update();
	_filled_lines->update();


	_c_pt_lines->update();
	_c_pt_approx_lines->update();
}
