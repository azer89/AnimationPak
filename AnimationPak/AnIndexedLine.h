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

#include "A3DVector.h"

#include "SystemParams.h"
#include <iostream>


//namespace CVSystem
//{
struct AnIndexedLine
{
public:
	A3DVector _dir;
	A3DVector _dir_cuda;

	float _mag; // for cuda debugging debug delete me
	float _mag_cuda; // for cuda debugging debug delete me

	float _diff; // for cuda debugging debug delete me
	float _diff_cuda; // for cuda debugging debug delete me

	float _k_debug; // for cuda debugging debug delete me
	float _k_debug_cuda; // for cuda debugging debug delete me

	float _signval_debug; // for cuda debugging debug delete me
	float _signval_debug_cuda; // for cuda debugging debug delete me

	float _dist;
	float _oriDist; // see MakeLonger()

	int _index0; // start index
	int _index1; // end index 
	float _scale;
	
	
	//bool _isLayer2Layer; // is this an inter-layer edge

public:
	void SetDist(float d)  { _dist = d; _oriDist = _dist; }
	//float GetDist() const  { return _dist; }
	//float GetScale() const { return _scale; }

public:

	// Constructor
	AnIndexedLine(int index0, int index1)
	{		
		this->_index0 = index0;
		this->_index1 = index1;
		this->_dist = 0.0f;
		this->_oriDist = 0.0f;		
		this->_scale = 1.0f;
		//this->_scale = 1.0f;
		//this->_angle = 0.0f;
		//this->_isLayer2Layer = isLayer2Layer;
		//this->_canGrow = true;
	}
	
	/*AnIndexedLine(int index0, int index1, float dist, float oriDist, bool isLayer2Layer = false)
	{
		//this->_scale = 1.0f;
		this->_index0 = index0;
		this->_index1 = index1;
		this->_dist = dist;
		this->_oriDist = oriDist;
		this->_isLayer2Layer = isLayer2Layer;
		//this->_angle = 0.0f;
		//this->_canGrow = true;
	}*/
	
	// SET THIS BEFORE SIMULATION OK!
	void SetActualOriDistance(float d)
	{
		this->_dist = d;
		this->_oriDist = d;
	}

	// debug delete me
	void DebugShit()
	{
		float mag_diff_abs = std::abs(_mag - _mag_cuda);
		//if (_dir != _dir_cuda)
		if(mag_diff_abs >= 1e-10)
		{
			//std::cout << "dist = " << std::abs(_diff - _diff_cuda) << "\n";
			std::cout << mag_diff_abs << "\n";
			std::cout << "dir = " << _dir.ToString() << "\n";
			std::cout << "dir_cuda = " << _dir_cuda.ToString() << "\n\n";
		}

		/*float diff_abs = std::abs(_diff - _diff_cuda);

		if(diff_abs > 1e-4)
		{
			std::cout << "d   " << _diff << ", " << _diff_cuda << "\n";
		}*/

		/*if ((_diff < 0 && _diff_cuda > 0) || (_diff > 0 && _diff_cuda < 0))
		{
			std::cout << "d   " << _diff << ", " << _diff_cuda << "\n";
			std::cout << "s   " << _signval_debug << ", " << _signval_debug_cuda << "\n\n";
		}*/
	}

	// debug delete me
	/*void PrintKCUDA()
	{
		float diff_abs = std::abs(_k_debug - _k_debug_cuda);

		if (diff_abs > 0.0)
		{
			std::cout << "k   " << _k_debug << ", " << _k_debug_cuda << "\n";
		}
	}*/

	// debug delete me
	/*void PrintSignValCUDA()
	{
		float diff_abs = std::abs(_signval_debug - _signval_debug_cuda);

		if (diff_abs > 0.0)
		{
			std::cout << "s   " << _signval_debug << ", " << _signval_debug_cuda << "\n";
			std::cout << "d   " << _diff << ", " << _diff_cuda << "\n\n";
		}
	}*/

	/*AnIndexedLine(const AMass& m1, const AMass& m2, bool isLayer2Layer = false)
	{
		this->_index0 = m1._self_idx;
		this->_index1 = m2._self_idx;
		
		A3DVector pt1 = m1._pos;
		A3DVector pt2 = m2._pos;

		this->_dist = pt1.Distance(pt2);
		this->_oriDist = this->_dist;
		this->_isLayer2Layer = isLayer2Layer;
	}*/
	
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
//}

#endif