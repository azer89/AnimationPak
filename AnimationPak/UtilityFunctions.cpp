
/* ---------- AnimationPak  ---------- */

#include "UtilityFUnctions.h"

#include "A2DVector.h"
//#include "ALine.h"
//#include "ARectangle.h"

#include <sstream>

#define PI 3.14159265359
#define PI2 6.28318530718


int UtilityFunctions::GetIndexFromIntList(const std::vector<int>& aList, int elem)
{
	for (unsigned int a = 0; a < aList.size(); a++)
	{
		if (elem == aList[a]) { return a; }
	}

	return -1;
}