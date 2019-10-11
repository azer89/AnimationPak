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

struct AnIndexedLine
{
public:
	float _dist;
	float _oriDist;

	int _index0; // start index
	int _index1; // end index 
	float _scale;
	
	int _valence; // the lowest valence between two vertices, see SolveSprings3D()
		
	bool _isLayer2Layer; // is this an inter-layer edge

public:
	void SetDist(float d)  { _dist = d; _oriDist = _dist; }

public:

	// Constructor
	AnIndexedLine(int index0, int index1, bool isLayer2Layer = false)
	{
		//this->_scale = 1.0f;
		this->_index0 = index0;
		this->_index1 = index1;
		this->_dist = 0.0f;
		this->_oriDist = 0.0f;
		//this->_angle = 0.0f;
		this->_isLayer2Layer = isLayer2Layer;
		//this->_canGrow = true;
		this->_scale = 1.0f;

		this->_valence = 0;
	}
	
	// SET THIS BEFORE SIMULATION OK!
	void SetActualOriDistance(float d)
	{
		this->_dist = d;
		this->_oriDist = d;
	}


	void MakeLonger(float growth_scale_iter, float dt)
	{
		_scale += growth_scale_iter * dt;
		_dist = _oriDist * _scale;
	}

	// http_://stackoverflow.com/questions/3647331/how-to-swap-two-numbers-without-using-temp-variables-or-arithmetic-operations
	void Swap()
	{
		_index0 ^= _index1;
		_index1 ^= _index0;
		_index0 ^= _index1;
	}
};

#endif