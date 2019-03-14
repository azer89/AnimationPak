
/* ---------- ShapeRadiusMatching V2  ---------- */

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

//#include "SystemParams.h"

/**
* Reza Adhitya Saputra
* radhitya@uwaterloo.ca
* February 2016
*/



/**
* A struct to represent:
*     1. A Point
*     2. A Vector (direction only)
*/
struct A3DVector
{
public:
	// x
	float _x;

	// y
	float _y;

	// z
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
		//this->radAngle = 0;
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

		return std::isinf(_x) || std::isnan(_x) || std::isinf(_y) || std::isnan(_y) || std::isinf(_z) || std::isnan(_z);
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

	// Normalize
	A3DVector Norm() // get the unit vector
	{
		float vlength = std::sqrt(_x * _x + _y * _y + _z * _z);

		//if (vlength == 0) { std::cout << "div by zero duh\n"; }
		if (vlength == 0) { return A3DVector(0, 0, 0); }

		return A3DVector(this->_x / vlength, this->_y / vlength, this->_z / vlength);
	}

	// Euclidean distance
	float Distance(const A3DVector& other)
	{
		float xDist = _x - other._x;
		float yDist = _y - other._y;
		float zDist = _z - other._z;
		return std::sqrt(xDist * xDist + yDist * yDist + zDist * zDist);
	}

	// Euclidean distance
	float Distance(float otherX, float otherY, , float otherZ)
	{
		float xDist = _x - otherX;
		float yDist = _y - otherY;
		float yDist = _z - otherZ;
		return std::sqrt(xDist * xDist + yDist * yDist + zDist * zDist);
	}

	// squared euclidean distance
	float DistanceSquared(const AVector& other)
	{
		float xDist = x - other.x;
		float yDist = y - other.y;
		return (xDist * xDist + yDist * yDist);
	}

	// squared euclidean distance
	float DistanceSquared(float otherX, float otherY)
	{
		float xDist = x - otherX;
		float yDist = y - otherY;
		return (xDist * xDist + yDist * yDist);
	}

	// operator overloading
	AVector operator+ (const AVector& other) { return AVector(x + other.x, y + other.y); }

	// operator overloading
	AVector operator- (const AVector& other) { return AVector(x - other.x, y - other.y); }
	bool operator== (const AVector& other)
	{
		return (abs(this->x - other.x) < std::numeric_limits<float>::epsilon() && abs(this->y - other.y) < std::numeric_limits<float>::epsilon());
	}

	// operator overloading
	bool operator!= (const AVector& other)
	{
		return (abs(this->x - other.x) > std::numeric_limits<float>::epsilon() || abs(this->y - other.y) > std::numeric_limits<float>::epsilon());
	}

	// operator overloading
	AVector operator+= (const AVector& other)
	{
		x += other.x;
		y += other.y;
		return *this;
	}

	// operator overloading
	AVector operator-= (const AVector& other)
	{
		x -= other.x;
		y -= other.y;
		return *this;
	}

	// operator overloading
	AVector operator/ (const float& val) { return AVector(x / val, y / val); }

	// operator overloading
	AVector operator* (const float& val) { return AVector(x * val, y * val); }

	// operator overloading
	AVector operator*= (const float& val)
	{
		x *= val;
		y *= val;
		return *this;
	}

	// operator overloading
	AVector operator/= (const float& val)
	{
		x /= val;
		y /= val;
		return *this;
	}

	// length of a vector
	float Length() { return sqrt(x * x + y * y); }

	// squared length of a vector
	float LengthSquared() { return x * x + y * y; }

	// dot product
	float Dot(const AVector& otherVector) { return x * otherVector.x + y * otherVector.y; }

	// cross product
	AVector Cross(const AVector& otherVector)
	{
		//u x v = u.x * v.y - u.y * v.x
		return AVector(x * otherVector.y, y * otherVector.x);
	}

	// linear dependency test
	bool IsLinearDependent(const AVector& otherVector)
	{
		float det = (this->x * otherVector.y) - (this->y * otherVector.x);
		if (det > -std::numeric_limits<float>::epsilon() && det < std::numeric_limits<float>::epsilon()) { return true; }
		return false;
	}

	// angle direction
	// not normalized
	A3DVector DirectionTo(const A3DVector& otherVector)
	{
		return A3DVector(otherVector._x - this->_x, otherVector._y - this->_y, otherVector._z - this->_z);
	}

	// TODO: reimplement this
	//bool IsOut()
	//{
	//	return (_x < 0 || _x > SystemParams::_upscaleFactor || _y < 0 || _y > SystemParams::_upscaleFactor || _z < 0 || _z > SystemParams::_upscaleFactor);
	//}

	void Print()
	{
		std::cout << "(" << _x << ", " << _y << ", " << _z << ")\n";
	}
};

//typedef std::vector<std::vector<AVector>> GraphArt;

#endif // AVECTOR_H