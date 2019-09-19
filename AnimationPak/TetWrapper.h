#ifndef __TET_WRAPPER_H
#define __TET_WRAPPER_H

#include "AnIndexedLine.h"

#include "A3DVector.h"
#include "AMass.h"

#include <vector>

class TetWrapper
{
public:

	TetWrapper();

	~TetWrapper();

	//void GenerateTet(const std::vector<AMass>& massList,	
	//				 float maxDistRandPt,
	//	             std::vector<AnIndexedLine>& tetEdges);

	//void PruneEdges(const std::vector<AMass>& massList,
	//	            const std::vector<AnIndexedLine>& tetEdges);

	bool ValidIndex(int idx);

public:
	int _massSize;
};

#endif // !__TET_WRAPPER_H

