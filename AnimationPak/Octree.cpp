
#include "Octree.h"

#include <algorithm>
#include <initializer_list>

Octree::Octree(float x, float y, float z, float side_width, int level, int max_level, Octree* parent)
{
	_x = x;
	_y = y;
	_z = z;
	//_side_width = side_width;
	
	_x_max = _x + side_width;
	_y_max = _y + side_width;
	_z_max = _z - side_width;
	
	
	
	_level = level;
	_max_level = max_level;
	_parent = parent;

	if (_level == _max_level) 
	{
		return;
	}

	float half_side_width = side_width * 0.5;

	/*
	seen from top

	(-z)
	^
	|
	| 3  4
	| 1  2
	---------> (+x)

	*/

	_top1 = new Octree(x,                   y + half_side_width, z,                   half_side_width, _level + 1, _max_level, this);
	_top2 = new Octree(x + half_side_width, y + half_side_width, z,                   half_side_width, _level + 1, _max_level, this);
	_top3 = new Octree(x,                   y + half_side_width, z - half_side_width, half_side_width, _level + 1, _max_level, this);
	_top4 = new Octree(x + half_side_width, y + half_side_width, z - half_side_width, half_side_width, _level + 1, _max_level, this);

	/*
	seen from top

	(-z)
	^
	|
	| 3  4
	| 1  2
	---------> (+x)

	*/

	_bottom1 = new Octree(x,                   y, z,                   half_side_width, _level + 1, _max_level, this);
	_bottom2 = new Octree(x + half_side_width, y, z,                   half_side_width, _level + 1, _max_level, this);
	_bottom3 = new Octree(x,                   y, z - half_side_width, half_side_width, _level + 1, _max_level, this);
	_bottom4 = new Octree(x + half_side_width, y, z - half_side_width, half_side_width, _level + 1, _max_level, this);
}


Octree::~Octree()
{
	if (_level == _max_level)
	{
		return;
	}

	delete _top1;
	delete _top2;
	delete _top3;
	delete _top4;

	delete _bottom1;
	delete _bottom2;
	delete _bottom3;
	delete _bottom4;
}

void Octree::AddObject(A3DObject* object)
{
	//_objects.push_back(obj);
	/*if (_level == _max_level)
	{
		_objects.push_back(object);
		return;
	}*/

	if (Contains(_top1, object))
	{
		_top1->AddObject(object);
		return;
	}
	else if (Contains(_top2, object))
	{
		_top2->AddObject(object);
		return;
	}
	else if (Contains(_top3, object))
	{
		_top3->AddObject(object);
		return;
	}
	else if (Contains(_top4, object))
	{
		_top4->AddObject(object);
		return;
	}
	else if (Contains(_bottom1, object))
	{
		_bottom1->AddObject(object);
		return;
	}
	else if (Contains(_bottom2, object))
	{
		_bottom2->AddObject(object);
		return;
	}
	else if (Contains(_bottom3, object))
	{
		_bottom3->AddObject(object);
		return;
	}
	else if (Contains(_bottom4, object))
	{
		_bottom4->AddObject(object);
		return;
	}
	else
	{
		// stays here, not leaf
		_objects.push_back(object);
	}
}

void Octree::AddTriangle(A3DVector p1, A3DVector p2, A3DVector p3, int info1, int info2)
{
	float x_min = std::min({ p1._x, p2._x, p3._x });
	float y_min = std::min({ p1._y, p2._y, p3._y });
	float z_min = std::max({ p1._z, p2._z, p3._z });

	float x_max = std::max({ p1._x, p2._x, p3._x });
	float y_max = std::max({ p1._y, p2._y, p3._y });
	float z_max = std::min({ p1._z, p2._z, p3._z });

	A3DObject* object = new A3DObject(x_min, y_min, z_min, info1, info2);
	object->SetMaxVals(x_max, y_max, z_max);

	if (Contains(this, object))
	{
		AddObject(object);
		return;
	}
}

std::vector<A3DObject*>  Octree::GetInvalidObjectsAndReassign()
{
	std::vector<A3DObject*> invalidObjects;


	if (_level < _max_level) // not leaf
	{
		{
			std::vector<A3DObject*> childReturnObjects = _top1->GetInvalidObjectsAndReassign();
			invalidObjects.insert(invalidObjects.end(), childReturnObjects.begin(), childReturnObjects.end());
		}
		{
			std::vector<A3DObject*> childReturnObjects = _top2->GetInvalidObjectsAndReassign();
			invalidObjects.insert(invalidObjects.end(), childReturnObjects.begin(), childReturnObjects.end());
		}
		{
			std::vector<A3DObject*> childReturnObjects = _top3->GetInvalidObjectsAndReassign();
			invalidObjects.insert(invalidObjects.end(), childReturnObjects.begin(), childReturnObjects.end());
		}
		{
			std::vector<A3DObject*> childReturnObjects = _top4->GetInvalidObjectsAndReassign();
			invalidObjects.insert(invalidObjects.end(), childReturnObjects.begin(), childReturnObjects.end());
		}
		{
			std::vector<A3DObject*> childReturnObjects = _bottom1->GetInvalidObjectsAndReassign();
			invalidObjects.insert(invalidObjects.end(), childReturnObjects.begin(), childReturnObjects.end());
		}
		{
			std::vector<A3DObject*> childReturnObjects = _bottom2->GetInvalidObjectsAndReassign();
			invalidObjects.insert(invalidObjects.end(), childReturnObjects.begin(), childReturnObjects.end());
		}
		{
			std::vector<A3DObject*> childReturnObjects = _bottom3->GetInvalidObjectsAndReassign();
			invalidObjects.insert(invalidObjects.end(), childReturnObjects.begin(), childReturnObjects.end());
		}
		{
			std::vector<A3DObject*> childReturnObjects = _bottom4->GetInvalidObjectsAndReassign();
			invalidObjects.insert(invalidObjects.end(), childReturnObjects.begin(), childReturnObjects.end());
		}

		// try to reassign to the neighbors
		for (int a = invalidObjects.size() - 1; a >= 0; a--)
		{
			if (Contains(this, invalidObjects[a]))
			{
				AddObject(invalidObjects[a]);
				invalidObjects.erase(invalidObjects.begin() + a);
			}
		}
	}

	// invalid here
	for (int a = _objects.size() - 1; a >= 0; a--)
	{
		if (!Contains(this, _objects[a]))
		{
			invalidObjects.push_back(_objects[a]);
			_objects.erase(_objects.begin() + a);
		}
	}

	return invalidObjects;
}

bool Octree::Contains(Octree *node, A3DObject *object)
{
	// point
	/*return (
		// x
		object->_x > node->_x &&
		object->_x + object->_x_width < node->_x + node->_side_width &&

		// y
		object->_y > node->_y &&
		object->_y + object->_y_width < node->_y + node->_side_width &&

		//z
		object->_z < node->_z &&
		object->_z - object->_z_width > node->_z - node->_side_width
		);*/

	return (
		// x
		object->_x     > node->_x &&
		object->_x_max < node->_x_max &&

		// y
		object->_y     > node->_y &&
		object->_y_max < node->_y_max &&

		//z
		object->_z     < node->_z &&  
		object->_z_max > node->_z_max
		);
}