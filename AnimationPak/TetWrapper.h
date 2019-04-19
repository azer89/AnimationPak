#ifndef __TET_WRAPPER_H
#define __TET_WRAPPER_H

#include "AnIndexedLine.h"

#include "A3DVector.h"

#include <vector>

class TetWrapper
{
public:

	TetWrapper();

	~TetWrapper();

	void GenerateTet(std::vector<A3DVector> input_points);

	
};

#endif // !__TET_WRAPPER_H

