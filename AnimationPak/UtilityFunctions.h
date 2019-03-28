/* ---------- AnimationPak  ---------- */

/*
================================================================================
Reza Adhitya Saputra
radhitya@uwaterloo.ca
================================================================================
*/

#ifndef UTILITY_FUNCTIONS_H
#define UTILITY_FUNCTIONS_H

#include <vector>
//#include "APath.h"
//#include "ABary.h"

// forward declaration
struct AVector;
struct ALine;
struct ARectangle;


class UtilityFunctions
{
public:
	static int GetIndexFromIntList(const std::vector<int>& aList, int elem);
};

#endif