

#ifndef __A_2D_Square__
#define __A_2D_Square__

#include "A2DObject.h"

#include <iostream>
#include <vector>

class A2DSquare
{
public:
	A2DSquare(float x, float y, float length) :
		_x(x),
		_y(y),
		_length(length)
	{
		float offVal = 1.0f;
		_x1 = _x + offVal;
		_y1 = _y + offVal;
		_x2 = _x - offVal + _length;
		_y2 = _y - offVal + _length;

		_xCenter = _x + (_length * 0.5);
		_yCenter = _y + (_length * 0.5);

		//_containerFlag = 0;
	}

	~A2DSquare()
	{
	}

public:
	float _x;
	float _y;
	float _length;

	float _xCenter;
	float _yCenter;

	/*
	0 be careful
	1 nope !!!
	*/
	//int _containerFlag;

	// drawing
	float _x1;
	float _y1;
	float _x2;
	float _y2;

	std::vector<A2DObject*>	_objects;

	inline bool Contains(A2DObject* obj)
	{
		return !(obj->_x < _x ||
			obj->_y < _y ||
			obj->_x > _x + _length ||
			obj->_y > _y + _length);
	}

	void Clear()
	{
		this->_objects.clear();
	}

};

#endif