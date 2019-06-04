/*
================================================================================
Reza Adhitya Saputra
radhitya@uwaterloo.ca
================================================================================
*/

#ifndef PATH_IO_H
#define PATH_IO_H

#include <vector>
#include <cstring>

#include "A2DVector.h"

class PathIO
{
public:
	PathIO();
	~PathIO();

	std::vector<std::vector<A2DVector>> LoadElement(std::string filename);

};

#endif
