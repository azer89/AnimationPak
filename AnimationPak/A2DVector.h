
/* ---------- AnimationPak  ---------- */

/*
================================================================================
Reza Adhitya Saputra
radhitya@uwaterloo.ca
================================================================================
*/

#ifndef A2DVECTOR_H
#define A2DVECTOR_H

#include <limits>
#include <cmath>
#include <iostream> // for abs (?)

#include <vector>

#include "SystemParams.h"

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
struct A2DVector
{
public:
	// x
	float x;

	// y
	float y;

	//float radAngle;

	// what is this?
	float aCertainInfo;

	// Default constructor
	A2DVector()
	{
		this->x = -1;
		this->y = -1;
		//this->radAngle = 0;
	}

	// Constructor
	A2DVector(float x, float y)
	{
		this->x = x;
		this->y = y;
		//this->radAngle = 0;
	}

	// Scale a point
	A2DVector Resize(float val)
	{
		A2DVector newP;
		newP.x = this->x * val;
		newP.y = this->y * val;
		return newP;
	}

	void SetInvalid()
	{
		this->x = -1;
		this->y = -1;
	}

	bool IsBad()
	{
		//if (Length() > 1000000) { std::cout << "huh?\n"; return true; }

		return std::isinf(x) || std::isnan(x) || std::isinf(y) || std::isnan(y);
	}

	// if a point is (-1, -1)
	bool IsInvalid()
	{
		if (((int)x) == -1 && ((int)y) == -1)
		{
			return true;
		}
		return false;
	}

	// Normalize
	A2DVector Norm() // get the unit vector
	{
		float vlength = std::sqrt(x * x + y * y);

		//if (vlength == 0) { std::cout << "div by zero duh\n"; }
		if (vlength == 0) { return A2DVector(0, 0); }

		return A2DVector(this->x / vlength, this->y / vlength);
	}

	// Euclidean distance
	float Distance(const A2DVector& other)
	{
		float xDist = x - other.x;
		float yDist = y - other.y;
		return std::sqrt(xDist * xDist + yDist * yDist);
	}

	void GetUnitAndDist(A2DVector& unitVec, float& dist)
	{

		dist = std::sqrt(x * x + y * y);

		unitVec = A2DVector(this->x / dist,
			this->y / dist);

	}

	// Euclidean distance
	float Distance(float otherX, float otherY)
	{
		float xDist = x - otherX;
		float yDist = y - otherY;
		return std::sqrt(xDist * xDist + yDist * yDist);
	}

	// squared euclidean distance
	float DistanceSquared(const A2DVector& other)
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
	A2DVector operator+ (const A2DVector& other) { return A2DVector(x + other.x, y + other.y); }

	// operator overloading
	A2DVector operator- (const A2DVector& other) { return A2DVector(x - other.x, y - other.y); }
	bool operator== (const A2DVector& other)
	{
		return (abs(this->x - other.x) < std::numeric_limits<float>::epsilon() && abs(this->y - other.y) < std::numeric_limits<float>::epsilon());
	}

	// operator overloading
	bool operator!= (const A2DVector& other)
	{
		return (abs(this->x - other.x) > std::numeric_limits<float>::epsilon() || abs(this->y - other.y) > std::numeric_limits<float>::epsilon());
	}

	// operator overloading
	A2DVector operator+= (const A2DVector& other)
	{
		x += other.x;
		y += other.y;
		return *this;
	}

	// operator overloading
	A2DVector operator-= (const A2DVector& other)
	{
		x -= other.x;
		y -= other.y;
		return *this;
	}

	// operator overloading
	A2DVector operator/ (const float& val) { return A2DVector(x / val, y / val); }

	// operator overloading
	A2DVector operator* (const float& val) { return A2DVector(x * val, y * val); }

	// operator overloading
	A2DVector operator*= (const float& val)
	{
		x *= val;
		y *= val;
		return *this;
	}

	// operator overloading
	A2DVector operator/= (const float& val)
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
	float Dot(const A2DVector& otherVector) { return x * otherVector.x + y * otherVector.y; }

	// cross product
	A2DVector Cross(const A2DVector& otherVector)
	{
		//u x v = u.x * v.y - u.y * v.x
		return A2DVector(x * otherVector.y, y * otherVector.x);
	}

	// linear dependency test
	bool IsLinearDependent(const A2DVector& otherVector)
	{
		float det = (this->x * otherVector.y) - (this->y * otherVector.x);
		if (det > -std::numeric_limits<float>::epsilon() && det < std::numeric_limits<float>::epsilon()) { return true; }
		return false;
	}

	// angle direction
	// not normalized
	A2DVector DirectionTo(const A2DVector& otherVector)
	{
		return A2DVector(otherVector.x - this->x, otherVector.y - this->y);
	}

	bool IsOut()
	{
		return (x < 0 || x > SystemParams::_upscaleFactor || y < 0 || y > SystemParams::_upscaleFactor);
	}

	void Print()
	{
		std::cout << "(" << x << ", " << y << ")\n";
	}
};

typedef std::vector<std::vector<A2DVector>> GraphArt;

#endif // AVECTOR_H