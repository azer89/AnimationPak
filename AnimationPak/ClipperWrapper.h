
/* ---------- ShapeRadiusMatching V2  ---------- */

/*
================================================================================
Reza Adhitya Saputra
radhitya@uwaterloo.ca
================================================================================
*/

#ifndef CLIPPER_WRAPPER_H
#define CLIPPER_WRAPPER_H

#include "A2DVector.h"
//#include "ALine.h"

//#include "AGraph.h"
//#include "VFRegion.h"

//struct VFRegion;
//struct AGraph;

#include <vector>

class ClipperWrapper
{
public:
	ClipperWrapper();
	~ClipperWrapper();

	// offsetting 
	static std::vector<std::vector<A2DVector>> RoundOffsettingP(std::vector<A2DVector> polygon, float offsetVal);  // closed poly	
	static std::vector<std::vector<A2DVector>> RoundOffsettingPP(std::vector<std::vector<A2DVector >> polygons, float offsetVal);  // closed polys


public:
	static float _cScaling;
};

#endif





