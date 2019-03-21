
#ifndef __A_MASS_H__
#define __A_MASS_H__

#include "A3DVector.h"
#include "AMass.h"

class AMass
{
public:

	float   _mass;    // is likely only one
	A3DVector _pos;	  // current
	A3DVector _velocity;

	int   _idx;


public:
	AMass();

	~AMass();

	// Constructor
	AMass(float x, float y, float z);

	// Constructor
	AMass(A3DVector pos);

	void CallMeFromConstructor();

	void Init(); // reset forces to zero
	void Simulate(float dt);
	void Solve(/*Need more parameters*/);

private:
	A3DVector _edgeForce;
	A3DVector _repulsionForce;
	A3DVector _boundaryForce;
	A3DVector _overlapForce;
	A3DVector _rotationForce;
};

#endif