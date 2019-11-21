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

// read directory
#include "dirent.h"

#include "A2DVector.h"
#include "AnElement.h"

class PathIO
{
public:
	PathIO();
	~PathIO();

	std::vector<std::vector<A2DVector>> LoadElement(std::string filename);

	AnElement LoadAnimatedElement(std::string filename);

	std::vector<std::string> LoadFiles(std::string directoryPath); // read directory

	// scripted initial placement
	void LoadPaths(std::vector <std::vector<A3DVector>>& paths,
				   std::vector<std::vector<int>> layer_indices,
		           std::vector<A2DVector>& positions, 
		           std::string filename);

};

#endif
