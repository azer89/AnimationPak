
/* ---------- AnimationPak  ---------- */

/*
================================================================================
Reza Adhitya Saputra
radhitya@uwaterloo.ca
================================================================================
*/

#ifndef A3DVECTOR_H
#define A3DVECTOR_H

#include <limits>
#include <cmath>
#include <iostream> // for abs (?)

#include <vector>
#include <sstream>
#include "cuda_runtime.h"

#include "A2DVector.h"

//#include "SystemParams.h"

/**
* Reza Adhitya Saputra
* radhitya@uwaterloo.ca
* March 2019
*/



/**
* A struct to represent:
*     1. A Point
*     2. A Vector (direction only)
*/
struct A3DVector
{
public:
	float _x;
	float _y;
	float _z;

	// Default constructor
	__host__ __device__
	A3DVector()
	{
		this->_x = -1;
		this->_y = -1;
		this->_z = -1;
	}

	// Constructor
	__host__ __device__
	A3DVector(float x, float y, float z)
	{
		this->_x = x;
		this->_y = y;
		this->_z = z;
	}

	// combine Norm() and Distance()
	__host__ __device__
	void GetUnitAndDist(A3DVector& unitVec, float& dist)
	{
		/*float isqrt = inv_sqrt(_x * _x + _y * _y + _z * _z);
		dist = 1.0 / isqrt;
		unitVec = A3DVector(this->_x * isqrt,
			this->_y * isqrt,
			this->_z * isqrt);*/
		

		// original
		
		dist = std::sqrt(_x * _x + _y * _y + _z * _z);

		unitVec =  A3DVector(this->_x / dist,
			this->_y / dist,
			this->_z / dist);
		
	}

	// Normalize
	__host__ __device__
	A3DVector Norm() // get the unit vector
	{
		float vlength = std::sqrt(_x * _x + _y * _y + _z * _z);

		if (vlength == 0) { return A3DVector(0, 0, 0); }

		return A3DVector(this->_x / vlength, 
			             this->_y / vlength, 
			             this->_z / vlength);
	}

	// Euclidean distance
	__host__ __device__
	float Distance(const A3DVector& other)
	{
		float xDist = _x - other._x;
		float yDist = _y - other._y;
		float zDist = _z - other._z;
		return std::sqrt(xDist * xDist + 
			             yDist * yDist + 
			             zDist * zDist);
	}

	// Euclidean distance
	__host__ __device__
	float Distance(float otherX, float otherY, float otherZ)
	{
		float xDist = _x - otherX;
		float yDist = _y - otherY;
		float zDist = _z - otherZ;
		return std::sqrt(xDist * xDist + 
			             yDist * yDist + 
			             zDist * zDist);
	}

	// squared euclidean distance
	__host__ __device__
	float DistanceSquared(const A3DVector& other)
	{
		float xDist = _x - other._x;
		float yDist = _y - other._y;
		float zDist = _z - other._z;
		return (xDist * xDist + 
			    yDist * yDist + 
			    zDist * zDist);
	}

	// squared euclidean distance
	__host__ __device__
	float DistanceSquared(float otherX, float otherY, float otherZ)
	{
		float xDist = _x - otherX;
		float yDist = _y - otherY;
		float zDist = _z - otherZ;
		return (xDist * xDist + yDist * yDist + zDist * zDist);
	}

	// operator overloading
	__host__ __device__
	A3DVector operator+ (const A3DVector& other) { return A3DVector(_x + other._x, _y + other._y, _z + other._z); }

	// operator overloading
	__host__ __device__
	A3DVector operator- (const A3DVector& other) { return A3DVector(_x - other._x, _y - other._y, _z - other._z); }

	// equal
	__host__ __device__
	bool operator== (const A3DVector& other)
	{
		return (abs(this->_x - other._x) < std::numeric_limits<float>::epsilon() && 
			    abs(this->_y - other._y) < std::numeric_limits<float>::epsilon() && 
			    abs(this->_z - other._z) < std::numeric_limits<float>::epsilon());
	}

	// operator overloading
	__host__ __device__
	bool operator!= (const A3DVector& other)
	{
		return (abs(this->_x - other._x) > std::numeric_limits<float>::epsilon() || 
			    abs(this->_y - other._y) > std::numeric_limits<float>::epsilon() || 
			    abs(this->_z - other._z) > std::numeric_limits<float>::epsilon());
	}

	// operator overloading
	__host__ __device__
	A3DVector operator+= (const A3DVector& other)
	{
		_x += other._x;
		_y += other._y;
		_z += other._z;
		return *this;
	}

	// operator overloading
	__host__ __device__
	A3DVector operator-= (const A3DVector& other)
	{
		_x -= other._x;
		_y -= other._y;
		_z -= other._z;
		return *this;
	}

	// operator overloading
	__host__ __device__
	A3DVector operator/ (const float& val) { return A3DVector(_x / val, _y / val, _z / val); }

	// operator overloading
	__host__ __device__
	A3DVector operator* (const float& val) { return A3DVector(_x * val, _y * val, _z * val); }

	// operator overloading
	__host__ __device__
	A3DVector operator*= (const float& val)
	{
		_x *= val;
		_y *= val;
		_z *= val;
		return *this;
	}

	// operator overloading
	__host__ __device__
	A3DVector operator/= (const float& val)
	{
		_x /= val;
		_y /= val;
		_z /= val;
		return *this;
	}

	// length of a vector
	__host__ __device__
	float Length() { return sqrt(_x * _x + 
		                         _y * _y + 
		                         _z * _z); }

	// squared length of a vector
	__host__ __device__
	float LengthSquared() { return _x * _x + 
		                           _y * _y + 
		                           _z * _z; }

	// dot product
	__host__ __device__
	float Dot(const A3DVector& otherVector) { return _x * otherVector._x + 
		                                             _y * otherVector._y + 
		                                             _z * otherVector._z; }

	// angle direction, not normalized
	__host__ __device__
	A3DVector DirectionTo(const A3DVector& otherVector)
	{
		return A3DVector(otherVector._x - this->_x,
						 otherVector._y - this->_y,
						 otherVector._z - this->_z);
	}

	A2DVector GetA2DVector()
	{
		return A2DVector(_x, _y);
	}
	
    std::string ToString()
	{
		//std::cout << "(" << _x << ", " << _y << ", " << _z << ")\n";
		std::stringstream ss;
		ss << "(" << _x << ", " << _y << ", " << _z << ")";
		return ss.str();
	}

	// if a point is (-1, -1)
	/*__host__ __device__
	bool IsInvalid()
	{
		if (((int)_x) == -1 && ((int)_y) == -1 && ((int)_z) == -1)
		{
			return true;
		}
		return false;
	}*/

	/*void SetXY(float x, float y)
	{
		_x = x;
		_y = y;
	}*/

	/*void SetPosition(const A3DVector& otherPt)
	{

		this->_x = otherPt._x;
		this->_y = otherPt._y;
		this->_z = otherPt._z;
	}*/

	// Scale a point (was Resize)
	/*A3DVector Rescale(float val)
	{
		A3DVector newP;
		newP._x = this->_x * val;
		newP._y = this->_y * val;
		newP._z = this->_z * val;
		return newP;
	}*/

	/*void SetInvalid()
	{
		this->_x = -1;
		this->_y = -1;
		this->_z = -1;
	}*/

	// angle direction
	// not normalized
	/*A3DVector DirectionTo(const A3DVector& otherVector)
	{
		return A3DVector(otherVector._x - this->_x,
						 otherVector._y - this->_y,
						 otherVector._z - this->_z);
	}*/

	// TODO: reimplement this
	// cross product
	/*AVector Cross(const AVector& otherVector)
	{
		//u x v = u.x * v.y - u.y * v.x
		return AVector(x * otherVector.y, y * otherVector.x);
	}*/

	// TODO: reimplement this
	// linear dependency test
	/*
	bool IsLinearDependent(const A3DVector& otherVector)
	{
		float det = (this->x * otherVector.y) - (this->y * otherVector.x);
		if (det > -std::numeric_limits<float>::epsilon() && det < std::numeric_limits<float>::epsilon()) { return true; }
		return false;
	}*/


	// TODO: reimplement this
	//bool IsOut()
	//{
	//	return (_x < 0 || _x > SystemParams::_upscaleFactor || _y < 0 || _y > SystemParams::_upscaleFactor || _z < 0 || _z > SystemParams::_upscaleFactor);
	//}


private:
	// the fuck http_//h14s.p5r.org/2012/09/0x5f3759df.html?mwh=1
	/*float inv_sqrt_do_not_use(float number)
	{
		const float x2 = number * 0.5F;
		const float threehalfs = 1.5F;

		union {
			float f;
			uint32_t i;
		} conv = { number }; // member 'f' set to value of 'number'.
		conv.i = 0x5f3759df - (conv.i >> 1);
		conv.f *= (threehalfs - (x2 * conv.f * conv.f));
		return conv.f;
	}*/

};


#endif // AVECTOR_H