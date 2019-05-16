

#ifndef __A_3D_Square__
#define __A_3D_Square__

#include "A3DObject.h"

#include <iostream>
#include <vector>

class A3DSquare
{
public:
	A3DSquare(float x, float y, float z, float length) :
		_x(x),
		_y(y),
		_z(z),
		_length(length)
	{
		//float offVal = 1.0f;
		//_x1 = _x + offVal;
		//_y1 = _y + offVal;
		//_x2 = _x - offVal + _length;
		//_y2 = _y - offVal + _length;

		_xCenter = _x + (_length * 0.5);
		_yCenter = _y + (_length * 0.5);
		_zCenter = _z + (_length * 0.5);

		//_containerFlag = 0;
	}

	~A3DSquare()
	{
	}

public:
	float _x;
	float _y;
	float _z;

	float _length;

	float _xCenter;
	float _yCenter;
	float _zCenter;
	/*
	0 be careful
	1 nope !!!
	*/
	//int _containerFlag;

	// drawing
	//float _x1;
	//float _y1;
	//float _x2;
	//float _y2;

	std::vector<A3DObject*>	_objects;

	inline bool Contains(A3DObject* obj)
	{
		return !(obj->_x < _x ||
			obj->_y < _y ||
			obj->_z < _z ||
			obj->_x > _x + _length ||
			obj->_y > _y + _length ||
			obj->_z > _z + _length);
	}

	void Clear()
	{
		this->_objects.clear();
	}

};

#endif