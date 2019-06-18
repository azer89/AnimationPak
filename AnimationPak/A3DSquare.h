

#ifndef __A_3D_Square__
#define __A_3D_Square__

#include "A3DObject.h"
#include "A3DVector.h"

#include <iostream>
#include <vector>


typedef std::vector<std::pair<int, int>> PairData; // 1 which element 2 which triangle


// The name 
class A3DSquare
{
public:
	A3DSquare(float x, float y, float z, float length) :
		_x(x),
		_y(y),
		_z(z),
		_length(length)
	{
		int sz = 5000;
		//_pd = PairData(sz, std::pair<int, int>(0, 0));
		//_approx_pd = PairData(sz, std::pair<int, int>(0, 0));
		_pd = new std::pair<int, int>[SystemParams::_max_exact_array_len];
		_approx_pd = new std::pair<int, int>[SystemParams::_max_approx_array_len];
		_pd_actual_size = 0;
		_approx_pd_actual_size = 0;

		//float offVal = 1.0f;
		//_x1 = _x + offVal;
		//_y1 = _y + offVal;
		//_x2 = _x - offVal + _length;
		//_y2 = _y - offVal + _length;

		_xCenter = _x + (_length * 0.5);
		_yCenter = _y + (_length * 0.5);
		_zCenter = _z + (_length * 0.5);

		//drawing
		/*
		Base

		3 - 4
		|   |
		1 - 2
		*/
		_draw_pt1 = A3DVector(_x,           _y,          _z);
		_draw_pt2 = A3DVector(_x + _length, _y,          _z);
		_draw_pt3 = A3DVector(_x,           _y,          _z + _length);
		_draw_pt4 = A3DVector(_x + _length, _y,          _z + _length);

		/*
		Top

		7 - 8
		|   |
		5 - 6
		*/
		_draw_pt5 = A3DVector(_x,           _y + _length, _z);
		_draw_pt6 = A3DVector(_x + _length, _y + _length, _z);
		_draw_pt7 = A3DVector(_x,           _y + _length, _z + _length);
		_draw_pt8 = A3DVector(_x + _length, _y + _length, _z + _length);
	}

	~A3DSquare()
	{
		delete[] _pd;
		delete[] _approx_pd;
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
	/*
	Base

	3 - 4
	|   |
	1 - 2
	*/
	A3DVector _draw_pt1;
	A3DVector _draw_pt2;
	A3DVector _draw_pt3;
	A3DVector _draw_pt4;
	

	/*
	Top

	7 - 8
	|   |
	5 - 6
	*/
	A3DVector _draw_pt5;
	A3DVector _draw_pt6;
	A3DVector _draw_pt7;
	A3DVector _draw_pt8;
	//float _x1;
	//float _y1;
	//float _x2;
	//float _y2;

	// temp
	int _pd_actual_size;
	int _approx_pd_actual_size;
	std::pair<int, int>* _pd; //  std::vector<std::pair<int, int>>
	std::pair<int, int>*  _approx_pd; //  std::vector<std::pair<int, int>>

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