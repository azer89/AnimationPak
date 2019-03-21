
#include "AMass.h"

AMass::AMass()
{
	//this->_m     = 0;             // mass is always one
	this->_pos = A3DVector(0, 0, 0);
	this->_idx = -1;
	//this->_cellIdx = -1;

	CallMeFromConstructor();
}

// Constructor
AMass::AMass(float x, float y, float z)
{
	this->_pos = A3DVector(x, y, z);
	this->_idx = -1;

	CallMeFromConstructor();
}

// Constructor
AMass::AMass(A3DVector pos)
{
	this->_pos = pos;
	this->_idx = -1;

	CallMeFromConstructor();
}

AMass::~AMass()
{
}

void AMass::CallMeFromConstructor()
{

}

void AMass::Init()
{
	//_attractionForce = AVector(0, 0);
	this->_edgeForce      = A3DVector(0, 0, 0);
	this->_repulsionForce = A3DVector(0, 0, 0);
	this->_boundaryForce  = A3DVector(0, 0, 0);
	this->_overlapForce   = A3DVector(0, 0, 0);
	this->_rotationForce  = A3DVector(0, 0, 0);
}

void AMass::Simulate(float dt)
{
}

void AMass::Solve(/*Need more parameters*/)
{
}

