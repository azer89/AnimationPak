
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
	   
	//float radAngle;

	// what is this?
	//float aCertainInfo;

	// Default constructor
	A3DVector()
	{
		SetInvalid();
		//this->radAngle = 0;
	}

	// Constructor
	A3DVector(float x, float y, float z)
	{
		this->_x = x;
		this->_y = y;
		this->_z = z;
	}

	// Scale a point (was Resize)
	A3DVector Rescale(float val)
	{
		A3DVector newP;
		newP._x = this->_x * val;
		newP._y = this->_y * val;
		newP._z = this->_z * val;
		return newP;
	}

	void SetInvalid()
	{
		this->_x = -1;
		this->_y = -1;
		this->_z = -1;
	}

	bool IsBad()
	{
		//if (Length() > 1000000) { std::cout << "huh?\n"; return true; }
		return std::isinf(_x) || std::isnan(_x) || 
			   std::isinf(_y) || std::isnan(_y) || 
			   std::isinf(_z) || std::isnan(_z);
	}

	// if a point is (-1, -1)
	bool IsInvalid()
	{
		if (((int)_x) == -1 && ((int)_y) == -1 && ((int)_z) == -1)
		{
			return true;
		}
		return false;
	}

	// combine Norm() and Distance()
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

	void SetPosition(const A3DVector& otherPt)
	{

		this->_x = otherPt._x;
		this->_y = otherPt._y;
		this->_z = otherPt._z;
	}

	// Normalize
	A3DVector Norm() // get the unit vector
	{
		float vlength = std::sqrt(_x * _x + _y * _y + _z * _z);

		//if (vlength == 0) { std::cout << "div by zero duh\n"; }
		if (vlength == 0) { return A3DVector(0, 0, 0); }

		return A3DVector(this->_x / vlength, 
			             this->_y / vlength, 
			             this->_z / vlength);
	}

	// Euclidean distance
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
	float DistanceSquared(float otherX, float otherY, float otherZ)
	{
		float xDist = _x - otherX;
		float yDist = _y - otherY;
		float zDist = _z - otherZ;
		return (xDist * xDist + yDist * yDist + zDist * zDist);
	}

	// operator overloading
	A3DVector operator+ (const A3DVector& other) { return A3DVector(_x + other._x, _y + other._y, _z + other._z); }

	// operator overloading
	A3DVector operator- (const A3DVector& other) { return A3DVector(_x - other._x, _y - other._y, _z - other._z); }
	bool operator== (const A3DVector& other)
	{
		return (abs(this->_x - other._x) < std::numeric_limits<float>::epsilon() && 
			    abs(this->_y - other._y) < std::numeric_limits<float>::epsilon() && 
			    abs(this->_z - other._z) < std::numeric_limits<float>::epsilon());
	}

	// operator overloading
	bool operator!= (const A3DVector& other)
	{
		return (abs(this->_x - other._x) > std::numeric_limits<float>::epsilon() || 
			    abs(this->_y - other._y) > std::numeric_limits<float>::epsilon() || 
			    abs(this->_z - other._z) > std::numeric_limits<float>::epsilon());
	}

	// operator overloading
	A3DVector operator+= (const A3DVector& other)
	{
		_x += other._x;
		_y += other._y;
		_z += other._z;
		return *this;
	}

	// operator overloading
	A3DVector operator-= (const A3DVector& other)
	{
		_x -= other._x;
		_y -= other._y;
		_z -= other._z;
		return *this;
	}

	// operator overloading
	A3DVector operator/ (const float& val) { return A3DVector(_x / val, _y / val, _z / val); }

	// operator overloading
	A3DVector operator* (const float& val) { return A3DVector(_x * val, _y * val, _z * val); }

	// operator overloading
	A3DVector operator*= (const float& val)
	{
		_x *= val;
		_y *= val;
		_z *= val;
		return *this;
	}

	// operator overloading
	A3DVector operator/= (const float& val)
	{
		_x /= val;
		_y /= val;
		_z /= val;
		return *this;
	}

	// length of a vector
	float Length() { return sqrt(_x * _x + 
		                         _y * _y + 
		                         _z * _z); }

	// squared length of a vector
	float LengthSquared() { return _x * _x + 
		                           _y * _y + 
		                           _z * _z; }

	// dot product
	float Dot(const A3DVector& otherVector) { return _x * otherVector._x + 
		                                             _y * otherVector._y + 
		                                             _z * otherVector._z; }

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

	// angle direction
	// not normalized
	A3DVector DirectionTo(const A3DVector& otherVector)
	{
		return A3DVector(otherVector._x - this->_x, 
			             otherVector._y - this->_y, 
			             otherVector._z - this->_z);
	}

	// TODO: reimplement this
	//bool IsOut()
	//{
	//	return (_x < 0 || _x > SystemParams::_upscaleFactor || _y < 0 || _y > SystemParams::_upscaleFactor || _z < 0 || _z > SystemParams::_upscaleFactor);
	//}

	A2DVector GetA2DVector()
	{
		return A2DVector(_x, _y);
	}

	void SetXY(float x, float y)
	{
		_x = x;
		_y = y;
	}

	void Print()
	{
		std::cout << "(" << _x << ", " << _y << ", " << _z << ")\n";
	}

private:
	// the fuck http_//h14s.p5r.org/2012/09/0x5f3759df.html?mwh=1
	float inv_sqrt_do_not_use(float number)
	{
		/*
		long i;
		float x2, y;
		const float threehalfs = 1.5F;

		x2 = number * 0.5F;
		y = number;
		i = *(long *)&y;                       // evil floating point bit level hacking
		i = 0x5f3759df - (i >> 1);               // what the fuck? 
		y = *(float *)&i;
		y = y * (threehalfs - (x2 * y * y));   // 1st iteration
		//	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

		return y;
		*/

		const float x2 = number * 0.5F;
		const float threehalfs = 1.5F;

		union {
			float f;
			uint32_t i;
		} conv = { number }; // member 'f' set to value of 'number'.
		conv.i = 0x5f3759df - (conv.i >> 1);
		conv.f *= (threehalfs - (x2 * conv.f * conv.f));
		return conv.f;
	}

};


#endif // AVECTOR_H