/**
*
* Line representation with index
*
* Author: Reza Adhitya Saputra (reza.adhitya.saputra@gmail.com)
* Version: 2019
*
*
*/

#ifndef ANINDEXEDLINE_H
#define ANINDEXEDLINE_H

#include "SystemParams.h"

//namespace CVSystem
//{
struct AnIndexedLine
{
public:
	int _index0; // start index

	int _index1; // end index 

	//bool  _canGrow;

	//float _angle;

	//float _scale;
	float _dist;
	float _oriDist;
	bool _isLayer2Layer; // is this an inter-layer edge

public:
	//void SetDist(float d) { _dist = d; _oriDist = _dist; }
	//float GetDist() const { return _dist; }
	//float GetScale() const { return _scale; }

public:

	// Constructor
	AnIndexedLine(int index0, int index1)
	{
		//this->_scale = 1.0f;

		this->_index0 = index0;
		this->_index1 = index1;
		this->_dist = 0.0f;
		this->_oriDist = 0.0f;
		//this->_angle = 0.0f;
		this->_isLayer2Layer = false;

		//this->_canGrow = true;
	}

	AnIndexedLine(int index0, int index1, float dist, float oriDist, bool isLayer2Layer = false)
	{
		//this->_scale = 1.0f;

		this->_index0 = index0;
		this->_index1 = index1;
		this->_dist = dist;
		this->_oriDist = oriDist;
		this->_isLayer2Layer = isLayer2Layer;
		//this->_angle = 0.0f;

		//this->_canGrow = true;
	}


	// Constructor
	/*AnIndexedLine(int index0, int index1, float dist)
	{
		this->_scale = 1.0f;

		this->_index0 = index0;
		this->_index1 = index1;
		this->_dist = dist;
		this->_angle = 0.0f;

		this->_canGrow = true;
	}*/

	// Construction without parameters
	/*AnIndexedLine()
	{
		this->_index0 = -1;
		this->_index1 = -1;
		this->_dist = 0;
		this->_angle = 0.0f;

		this->_maxDist = 0;
		this->_canGrow = true;
	}*/

	/*void MakeLonger(float growth_scale_iter, float dt)
	{
		_scale += growth_scale_iter * dt;
		_dist = _oriDist * _scale;
	}*/

	// http_://stackoverflow.com/questions/3647331/how-to-swap-two-numbers-without-using-temp-variables-or-arithmetic-operations
	void Swap()
	{
		_index0 ^= _index1;
		_index1 ^= _index0;
		_index0 ^= _index1;
	}
};
//}

#endif